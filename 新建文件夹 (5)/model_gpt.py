import torch
import torch.nn as nn
from transformers import AutoTokenizer
from dataclasses import dataclass
import torch.nn.functional as F
import warnings
from peft import LoraConfig, get_peft_model

from language_models import GPT2LMHeadModel, Qwen3ForCausalLM
from utils import enlarge_causal_mat, combine_attn_mask

@dataclass
class MultimodalConfig:
    base_model: str = "gpt2"  # Hugging Face model name
    num_fmri_tokens: int = 2250  # Maximum sequence length for modality input
    fmri_vocab_size: int = 8192  # Vocabulary size for modality tokens
    dropout: float = 0.0  # Dropout rate
    num_rois: int = 450

    use_fmri_lm_head: bool = True
    add_fmri_delimiter: bool = False

    # for instruction tuning
    peft_tune: bool = False  # Whether to apply PEFT (e.g., LoRA)
    freeze_pretrained_lora: bool = False  # Whether to freeze pretrained LoRA and add new LoRA on top
    use_cls_head: bool = False  # Whether to use a classification head instead of LM head and generate tokens
    num_classes: int = 2  # Number of classes for classification head

    temperature: float = 1
    top_p: float = None
    top_k: int = None

class MultimodalLLM(nn.Module):
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        
        # Load the base Hugging Face model and tokenizer
        if config.base_model == 'gpt2':
            self.base_model = GPT2LMHeadModel.from_pretrained('gpt2')
        elif 'qwen' in config.base_model.lower():
            self.base_model = Qwen3ForCausalLM.from_pretrained(config.base_model, trust_remote_code=True,
                torch_dtype=torch.bfloat16)
        else:
            raise ValueError(f"Base model {config.base_model} not supported yet")
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        
        try:
            self.n_embd = self.base_model.config.n_embd
        except:
            self.n_embd = self.base_model.config.hidden_size

        if config.peft_tune:  # apply lora
            self.apply_lora()
        
        if config.use_cls_head:  # use a classifier head on the final hidden states instead of doing autoregressive 
            print("Using classification head instead of LM head")
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.cls_head = nn.Linear(self.n_embd, config.num_classes)

        if config.add_fmri_delimiter:
            self.fmri_start = '<fmri>'
            self.fmri_end = '</fmri>'

        # Get vocab size and embedding dim from base model
        base_config = self.base_model.config
        self.original_vocab_size = base_config.vocab_size
        self.vocab_size = base_config.vocab_size + config.fmri_vocab_size

        # TODO here only resize the lm_head; consider removing this if we directly align fMRI tokenizer with text space,
        # and use the paired data to do fine-tuning (loss only computed on text tokens so no need to produce fMRI tokens)
        if config.use_fmri_lm_head:
            self._add_fmri_lm_head()
        else:
            self._resize_output_layer()

        # Setup special tokens for models that need them
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("Set pad_token = eos_token")
            else:
                num_added = self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                if num_added > 0:
                    print(f"Added {num_added} special tokens")
        
        self.drop = nn.Dropout(config.dropout)

        # Enable gradient checkpointing for memory efficiency
        # if hasattr(self.base_model, 'gradient_checkpointing_enable'):
        #     self.base_model.gradient_checkpointing_enable()
        #     print("✅ Gradient checkpointing enabled for memory efficiency")

        print(f"Loaded base model: {config.base_model}")
        print(f"Number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def apply_lora(self, adapter_name="default"):
        """Apply LoRA to the base model
        
        Args:
            adapter_name: Name for the LoRA adapter (default: "default")
        """
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # target_modules = [ "q_proj", "k_proj"]
        if 'qwen' in self.config.base_model.lower():
            # peft_config = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05, target_modules=target_modules)
            peft_config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.1, target_modules=target_modules)
        else:
            peft_config = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05)
        
        # Check if this is the first adapter
        if not hasattr(self.base_model, 'peft_config'):
            # First adapter - use get_peft_model
            self.base_model = get_peft_model(self.base_model, peft_config)
            print(f"Applied LoRA adapter '{adapter_name}' to the base model (first adapter)")
        else:
            # Additional adapter - use add_adapter
            self.base_model.add_adapter(adapter_name, peft_config)
            self.base_model.set_adapter(adapter_name)
            print(f"Added LoRA adapter '{adapter_name}' to the base model")
    
    def freeze_lora_adapter(self, adapter_name=None):
        """Freeze LoRA adapter parameters
        
        Args:
            adapter_name: Name of adapter to freeze. If None, freezes all current LoRA parameters.
        """
        if not hasattr(self.base_model, 'peft_config'):
            print("Warning: No LoRA adapters found to freeze")
            return
        
        frozen_count = 0
        for name, param in self.base_model.named_parameters():
            # Freeze parameters that contain 'lora' in their name
            if 'lora' in name.lower():
                # If adapter_name specified, only freeze that adapter's parameters
                if adapter_name is None or adapter_name in name:
                    param.requires_grad = False
                    frozen_count += 1
        
        if adapter_name:
            print(f"Froze {frozen_count} parameters in LoRA adapter '{adapter_name}'")
        else:
            print(f"Froze {frozen_count} LoRA parameters")
    
    def save_pretrained_lora(self, save_directory):
        """Save only the LoRA adapter weights"""
        if hasattr(self.base_model, 'save_pretrained'):
            self.base_model.save_pretrained(save_directory)
            print(f"LoRA adapter saved to {save_directory}")
        else:
            print("Warning: base_model does not have save_pretrained method (LoRA not applied?)")
    
    def load_lora_adapter(self, adapter_path):
        """Load LoRA adapter weights into the model"""
        from peft import PeftModel
        if not hasattr(self.base_model, 'peft_config'):
            print("Warning: Attempting to load LoRA adapter but model doesn't have LoRA applied. Applying LoRA first...")
            self.apply_lora()
        
        # Load adapter weights
        self.base_model = PeftModel.from_pretrained(self.base_model, adapter_path)
        print(f"LoRA adapter loaded from {adapter_path}")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _resize_embeddings(self):
        """Universally resize token embeddings (input embedding and lm_head)"""
        self.base_model.resize_token_embeddings(self.vocab_size)
        print(f"✅ Used model.resize_token_embeddings() for {self.base_model.__class__.__name__}")
    
    def _resize_output_layer(self):
        """Resize output projection layer (lm_head) across different architectures"""
        # Different models have different output layer names
        output_layer_names = ['lm_head', 'embed_out', 'output_projection', 'cls']
        
        for layer_name in output_layer_names:
            if hasattr(self.base_model, layer_name):
                old_layer = getattr(self.base_model, layer_name)
                if isinstance(old_layer, nn.Linear):
                    new_layer = nn.Linear(
                        old_layer.in_features,
                        self.vocab_size,
                        bias=old_layer.bias is not None,
                        dtype=old_layer.weight.dtype
                    )
                    
                    # Copy existing weights
                    with torch.no_grad():
                        new_layer.weight[:self.original_vocab_size] = old_layer.weight
                        nn.init.normal_(
                            new_layer.weight[self.original_vocab_size:], 
                            mean=0.0, 
                            std=0.02
                        )
                        if new_layer.bias is not None and old_layer.bias is not None:
                            new_layer.bias[:self.original_vocab_size] = old_layer.bias
                            nn.init.zeros_(new_layer.bias[self.original_vocab_size:])
                    
                    if layer_name == 'lm_head' and hasattr(self.base_model, 'wte'):  # keep original weight tying
                        new_layer.weight[:self.original_vocab_size] = self.base_model.wte.weight

                    setattr(self.base_model, layer_name, new_layer)
                    print(f"✅ Resized {layer_name} layer")
                    break
        else:
            warnings.warn("Could not find output projection layer to resize")

    def _add_fmri_lm_head(self):
        print("Adding a separate fMRI lm_head")
        self.fmri_lm_head = nn.Linear(self.n_embd, self.config.fmri_vocab_size, bias=False)
        nn.init.normal_(self.fmri_lm_head.weight, mean=0.0, std=0.02)        

    def _prepare_inputs(self, x_fmri=None, y_fmri=None, x_text=None, y_text=None, attention_mask=None):
        """Prepare and validate inputs for forward pass"""
        if x_fmri is not None:
            # 获取大模型参数（权重）的实际数据类型（比如 BFloat16）
            target_dtype = next(self.base_model.parameters()).dtype
            # 强制转换脑特征，防止后续矩阵乘法时 Float32 与 BFloat16 冲突
            x_fmri = x_fmri.to(dtype=target_dtype)
        pos_ids = None
        if x_fmri is not None and x_text is None:
            # fMRI-only input
            assert x_fmri.shape[1] <= self.config.num_fmri_tokens, \
                f"fMRI sequence length {x_fmri.shape[1]} exceeds max {self.config.num_fmri_tokens}"
            
            x = x_fmri
            assert attention_mask.ndim == 3
            attention_mask = attention_mask.unsqueeze(1)

            if hasattr(self, 'fmri_lm_head'):
                targets = y_fmri if y_fmri is not None else None
            else:
                targets = y_fmri + self.original_vocab_size if y_fmri is not None else None
            
            # TODO pos_ids computed for time_indices; consider remove it or consider the `spatiotemporal`
            pos_ids = torch.arange(0, x_fmri.shape[1]//self.config.num_rois, dtype=torch.long).repeat_interleave(self.config.num_rois).to(x_fmri.device)
            pos_ids = pos_ids.unsqueeze(0).expand(x_fmri.shape[0], -1)  # (batch_size, seq_len)

        elif x_text is not None and x_fmri is None:
            # Text-only input
            x = self.base_model.get_input_embeddings()(x_text)
            targets = y_text

            if hasattr(x_text, 'attention_mask'):  # padding mask, make it 4D
                raise NotImplementedError
            elif attention_mask == 'causal':  # allow sdpa do causal attention automatically
                attention_mask = None
                pos_ids = torch.arange(0, x.size(1), dtype=torch.long).to(x.device)
                pos_ids = pos_ids.unsqueeze(0).expand(x.shape[0], -1)  # (batch_size, seq_len)
            elif attention_mask is not None:  # attention mask is given as 2D (batch_size, seq_len) for padding mask
                assert attention_mask.ndim == 2
                # transform to 4D mask
                causal_mask = torch.tril(torch.ones((x.shape[1], x.shape[1]), device=x.device)).bool()
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)
                padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
                attention_mask = causal_mask & padding_mask  # (batch_size, 1, seq_len, seq_len)
                
                # Create position IDs that account for padding
                # For left padding: padding tokens should all have position 0, real tokens start from 0
                pos_ids = attention_mask.sum(dim=-1).squeeze(1).squeeze(1) - 1  # (batch_size, seq_len)
                # Clamp to ensure padding positions get 0
                pos_ids = pos_ids.clamp(min=0)                

        elif x_text is not None and x_fmri is not None:
            # Multimodal input: concatenate fMRI and text
            assert x_fmri.shape[1] <= self.config.num_fmri_tokens, \
                f"fMRI sequence length {x_fmri.shape[1]} exceeds max {self.config.num_fmri_tokens}"
            
            text_ids = x_text.input_ids if hasattr(x_text, 'input_ids') else x_text
            x_text = self.base_model.get_input_embeddings()(text_ids)

            pos_ids = torch.arange(0, x_fmri.shape[1]//self.config.num_rois, dtype=torch.long).repeat_interleave(self.config.num_rois).to(x_fmri.device)

            if self.config.add_fmri_delimiter:
                fmri_start, fmri_end = self.tokenizer.encode(self.fmri_start, return_tensors='pt').to(x_fmri.device), self.tokenizer.encode(self.fmri_end, return_tensors='pt').to(x_fmri.device)
                fmri_start, fmri_end = self.base_model.get_input_embeddings()(fmri_start), self.base_model.get_input_embeddings()(fmri_end)
                len_start, len_end = fmri_start.size(1), fmri_end.size(1)
                fmri_start_mask, fmri_end_mask = torch.tril(torch.ones(len_start, len_start)).unsqueeze(0).repeat(x_fmri.size(0), 1, 1).to(x_fmri.device), torch.tril(torch.ones(len_end, len_end)).unsqueeze(0).repeat(x_fmri.size(0), 1, 1).to(x_fmri.device)
                
                # adjust attention_mask first
                fmri_mask = torch.vmap(torch.block_diag, in_dims=(0, 0, 0))(fmri_start_mask, attention_mask[:, :x_fmri.size(1), :x_fmri.size(1)], fmri_end_mask)
                fmri_mask[:, len_start:, :len_start] = 1
                fmri_mask[:, -len_end:, :-len_end] = 1
                attention_mask = combine_attn_mask(fmri_mask, attention_mask[:, x_fmri.size(1):, x_fmri.size(1):])
                
                # add the tag tokens to x_fmri and adjust y_fmri
                x_fmri = torch.cat([fmri_start.repeat(x_fmri.size(0), 1, 1), x_fmri, fmri_end.repeat(x_fmri.size(0), 1, 1)], dim=1)
                if y_fmri is not None: y_fmri = torch.cat([torch.full((y_fmri.size(0), len_start), -1 - self.original_vocab_size, dtype=y_fmri.dtype).to(y_fmri.device), y_fmri, torch.full((y_fmri.size(0), len_end), -1 - self.original_vocab_size, dtype=y_fmri.dtype).to(y_fmri.device)], dim=1)
                
                # adjust pos_ids
                pos_ids += len_start
                pos_ids = torch.cat([torch.arange(0, len_start, dtype=torch.long).to(x_fmri.device), 
                                     pos_ids, 
                                     torch.arange(pos_ids.max()+1, pos_ids.max()+1+len_end, dtype=torch.long).to(x_fmri.device)], 
                                    dim=-1)

            text_pos_ids = torch.arange(0, x_text.size(1), dtype=torch.long).to(x_fmri.device)
            pos_ids = torch.cat([pos_ids, text_pos_ids + pos_ids.max() + 1], dim=-1)
            pos_ids = pos_ids.unsqueeze(0).expand(x_fmri.shape[0], -1)  # (batch_size, seq_len)

            # Concatenate: [fMRI_tokens] + [text_tokens]
            x = torch.cat([x_fmri, x_text], dim=1)
        
            # ======================================================================
            # 🚀【修复核心：构建 4D 广播级掩码，支持 GQA 的 8头 与 16头 物理对齐】
            # ======================================================================
            if 'qwen' in self.config.base_model.lower() or 'llama' in self.config.base_model.lower():
                total_len = x.size(1)
                
                # 1. 建立标准的 2D 因果掩码矩阵
                causal_2d = torch.tril(torch.ones((total_len, total_len), device=x.device)).bool()
                
                # 2. 建立多模态 padding 掩码: fMRI部分全 1 有效，Text部分根据是否为 pad 决定
                padding_mask_2d = torch.cat([
                    torch.ones((x_fmri.size(0), x_fmri.size(1)), device=x.device, dtype=torch.bool),
                    (text_ids != self.tokenizer.pad_token_id)
                ], dim=1) # 形状: [Batch, Total_Seq_Len]

                # 3. 组合因果关系与Padding，强行升维至标准的 4D 掩码: [Batch, 1, Seq_Len, Seq_Len]
                # 关键：Head 维设为 1。由于 PyTorch 广播机制，它面对 16头和 8头 时都可以自动完美复制，不会撞维报错。
                attention_mask = causal_2d.unsqueeze(0).unsqueeze(1) & padding_mask_2d.unsqueeze(1).unsqueeze(2)
                attention_mask = attention_mask.bool()
            else:
                # 如果是传统的 GPT2 等老模型，保留你原本的 3D/4D 处理
                assert attention_mask is not None and attention_mask.ndim == 3, "Must provide a 3D attention_mask for multimodal input"
                attention_mask = attention_mask.unsqueeze(1)

            if y_fmri is None and y_text is None:
                targets = None
            elif y_fmri == 'nan' and y_text == 'nan':
                targets = 'nan'
            else:
                assert y_fmri.max() == -1, \
                    "No loss should be computed on fMRI tokens during instruction tuning"
                targets = torch.cat((y_fmri, y_text), dim=-1)
        else:
            raise ValueError("Must provide at least one input type (x_fmri or x_text)")
        
        return x, attention_mask, targets, pos_ids

    def forward(self, x_fmri=None, y_fmri=None, x_text=None, y_text=None, attention_mask=None, lm_head=True, Y=None):
        """Y is only used with classification head for loss computation"""
        x, mask, targets, pos_ids = self._prepare_inputs(x_fmri, y_fmri, x_text, y_text, attention_mask)
        
        x = self.drop(x)
        
       
        is_output_hidden_stats = (not lm_head) or hasattr(self, 'fmri_lm_head') or hasattr(self, 'cls_head')
        
        outputs = self.base_model(inputs_embeds=x, attention_mask=mask, position_ids=pos_ids, output_hidden_states=is_output_hidden_stats)

        if not lm_head:
            return outputs.hidden_states[-1], None, None  # Return last hidden state if lm_head is False
        
        if hasattr(self, 'fmri_lm_head') and x_fmri is not None and x_text is None:  # fMRI -> fMRI
            logits = self.fmri_lm_head(outputs.hidden_states[-1][:, :x_fmri.size(1), :])
        elif hasattr(self, 'cls_head') and (x_fmri is not None and x_text is not None):
            ans_idx = (x_text == 23998).nonzero(as_tuple=True)[1]
            assert ans_idx.numel() == x_fmri.size(0), "Each sample must have one ' Answer' token"

            batch_size = x_fmri.size(0)
            hidden_states = outputs.hidden_states[-1] # [B, T, H]
            
            # 🚀 改进：为每个样本动态提取 Answer 之前的序列
            pooled_features = []
            for i in range(batch_size):
                # 提取该样本从 fMRI 区域到 Answer 标记的完整部分
                end_pos = x_fmri.size(1) + ans_idx[i]
                sample_hidden = hidden_states[i, :end_pos, :] # [T_i, H]
                # 对该样本做平均池化，不依赖 min()
                pooled_features.append(sample_hidden.mean(dim=0))
            
            logits = torch.stack(pooled_features) # [B, H]
            logits = self.cls_head(logits) # [B, 2]
            
            # ======================================================================
            # 🎯【核心修复】：彻底废除错误的 MSE 回归损失，回归纯正的加权分类交叉熵
            # ======================================================================
            if Y is not None:
                # 展平分类标签
                target_labels = Y.long().view(-1)
                
                # 为了防止你数据集中正常人（0）太多，抑郁症（1）太少导致的盲猜
                # 我们动态赋予抑郁症患者更多的损失权重（这里暂定 5.0 倍，可根据需要调整）
                class_weights = torch.tensor([1.0, 5.0], device=logits.device)
                
                # 强力激活二分类交叉熵监督信号
                loss = F.cross_entropy(logits, target_labels, weight=class_weights)
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

            # ======================================================================
            # 🎯【核心修复】：废除硬编码的 0，现场计算并返回真实的 Batch 准确率！
            # ======================================================================
            with torch.no_grad():
                preds = logits.argmax(dim=-1) # [Batch_Size]
                correct = (preds == Y.long().view(-1)).float()
                accuracy = correct.mean().item() # 真实的当前 batch 准确率
            
            return logits, loss, accuracy  # 🌟 成功交出真实准确率！
    
    @torch.no_grad()
    def generate(self, x_fmri=None, x_text=None, attention_mask=None, max_new_tokens=50, 
             temperature=1.0, top_k=None, top_p=None, do_sample=True, 
             pad_token_id=None, eos_token_id=None, allowed_tokens=None, **kwargs):
        """
        Generate sequences using the multimodal model.
        """
        self.eval()
        
        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        # Store original fMRI length before _prepare_inputs (which may add delimiters)
        original_fmri_len = x_fmri.shape[1] if x_fmri is not None else 0
        
        # print("generate num_fmri_tokens =", self.config.num_fmri_tokens)
        # print("generate x_fmri len =", x_fmri.shape[1])

        if x_fmri is not None:
            self.config.num_fmri_tokens = x_fmri.shape[1]
        # Prepare initial inputs
        x, mask, _, pos_ids = self._prepare_inputs(x_fmri=x_fmri, x_text=x_text, attention_mask=attention_mask)
        
        batch_size, initial_seq_len = x.shape[:2]
        device = x.device

        # Calculate actual fMRI length after _prepare_inputs (includes delimiters if enabled)
        if x_fmri is not None and x_text is not None:
            if self.config.add_fmri_delimiter:
                # Delimiters were added in _prepare_inputs
                fmri_start_len = len(self.tokenizer.encode(self.fmri_start))
                fmri_end_len = len(self.tokenizer.encode(self.fmri_end))
                actual_fmri_len = original_fmri_len + fmri_start_len + fmri_end_len
            else:
                actual_fmri_len = original_fmri_len
        else:
            actual_fmri_len = 0

        # Handle case when prompt is right-padded; move the padding to the left of fMRI
        if x_fmri is not None and x_text is not None:
            pad_mask = (x_text != pad_token_id).long()
            pad_mask[:, 0] = 1  # Ensure the first token is not masked (due to BOS == PAD)
            if not pad_mask.all().item():  # if we have padding tokens
                last_non_pad_idx = pad_mask.shape[1] - torch.argmax(pad_mask.flip(dims=[1]).int(), dim=1) - 1  # index of last non-pad token for each sample
                pad_len = x_text.size(1) - 1 - last_non_pad_idx  # number of padding tokens at the end
                
                # Shift embeddings to move padding from right to left
                indices = torch.arange(x.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
                shifted_indices = (indices - pad_len.unsqueeze(1)) % x.shape[1]
                x = torch.gather(x, 1, shifted_indices.unsqueeze(-1).expand_as(x))

                # Update pos_ids using the actual fMRI length (including delimiters)
                pos_mask = torch.arange(x_text.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
                pos_mask = pos_mask > last_non_pad_idx.unsqueeze(1)
                masked_pos_ids = pos_ids[:, actual_fmri_len:].masked_fill(pos_mask, 0)
                pos_ids = pos_ids.clone()
                pos_ids[:, actual_fmri_len:] = masked_pos_ids
                pos_ids = torch.gather(pos_ids, 1, shifted_indices)
                
                # Shift the attention mask in both dimensions (rows and columns).
                mask = torch.gather(mask, 2, shifted_indices.unsqueeze(1).unsqueeze(3).expand(-1, mask.size(1), -1, mask.size(3)))
                mask = torch.gather(mask, 3, shifted_indices.unsqueeze(1).unsqueeze(2).expand(-1, mask.size(1), mask.size(2), -1))

        # For text-only input, we can use the base model's generate directly
        if x_fmri is None and x_text is not None:
            text_ids = x_text.input_ids if hasattr(x_text, 'input_ids') else x_text
            text_attention_mask = (text_ids != pad_token_id).long()
            text_attention_mask[:, 0] = 1  # Ensure the first token is not masked (due to BOS == PAD)
            return self.base_model.generate(
                input_ids=text_ids, 
                attention_mask=text_attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **kwargs
            )
        
        # Initialize with existing sequence (for multimodal case, this includes text tokens)
        if x_text is not None:
            text_ids = x_text.input_ids if hasattr(x_text, 'input_ids') else x_text
            current_ids = text_ids.clone()
        else:
            # For fMRI-only, start with empty sequence or special token
            current_ids = torch.full((batch_size, 1), pad_token_id, dtype=torch.long, device=device)
        
        # First forward pass to get initial cache
        outputs = self.base_model(
            inputs_embeds=x,
            attention_mask=mask,
            position_ids=pos_ids,
            use_cache=True
        )
        past_key_values = outputs.past_key_values

        for step in range(max_new_tokens):
            # Prepare inputs for current step
            if step == 0:
                # First step: use full multimodal input
                logits = outputs.logits[:, -1, :self.original_vocab_size]
            else:
                # Subsequent steps: only use new text token embeddings
                last_token_embeds = self.base_model.get_input_embeddings()(current_ids[:, -1:])

                # Create attention mask for new token (can attend to all previous)
                new_attention_mask = torch.ones((batch_size, 1, 1, initial_seq_len + step), 
                                            device=device, dtype=x.dtype)
                
                # Update position IDs
                if pos_ids is not None:
                    new_pos_id = pos_ids.max().item() + step
                    current_pos_ids = torch.full((batch_size, 1), new_pos_id, device=device, dtype=pos_ids.dtype)
                else:
                    current_pos_ids = None
                    
                # Forward pass
                outputs = self.base_model(
                    inputs_embeds=last_token_embeds,
                    attention_mask=new_attention_mask,
                    position_ids=current_pos_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
                logits = outputs.logits[:, -1, :self.original_vocab_size]  # Get logits for last token, only keep text tokens
                past_key_values = outputs.past_key_values

            # Create a mask for allowed tokens
            if allowed_tokens is not None:
                allowed_mask = torch.full(logits.shape, float('-inf'), device=device)
                if isinstance(allowed_tokens, dict):
                    step_tokens = allowed_tokens.get(step)
                    if step_tokens is None:
                        step_tokens = allowed_tokens.get("default", None)
                    if step_tokens is None:
                        allowed_mask.zero_()  # stop constraining after we run out of explicit steps
                    else:
                        for token_list in step_tokens:
                            allowed_mask[:, token_list] = 0
                else:
                    for token_list in allowed_tokens:
                        allowed_mask[:, token_list] = 0
                logits = logits + allowed_mask
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample or select next token
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to current sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return current_ids
    
    def generate_text(self, x_fmri=None, x_text=None, attention_mask=None, 
                    max_new_tokens=50, **generate_kwargs):
        """
        Generate text and decode to strings.
        """
        generated_ids = self.generate(
            x_fmri=x_fmri, 
            x_text=x_text, 
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs
        )
        
        # Decode only the text tokens (filter out fMRI tokens if present)
        text_ids = generated_ids.clone()
        text_ids[text_ids >= self.original_vocab_size] = self.tokenizer.pad_token_id
        
        generated_texts = [self.tokenizer.decode(text_ids[i], skip_special_tokens=True) for i in range(text_ids.size(0))]
        return generated_texts