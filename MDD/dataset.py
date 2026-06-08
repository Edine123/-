import random
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os

from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from transformers import AutoTokenizer


class fMRITextDataset(Dataset):
    """
    fMRI-文本对齐数据集（已进行极致内存加速优化与类型兼容适配）
    """
    def __init__(
            self,
            file,
            inds=None,
            descriptor_types=None,
            lm_name="/root/MDD/gpt2",
            norm="robust",
            clip_timepoints=160,
            max_len=512,
            GPT_training=False,
            is_val=False,
            **kwargs):
        
        self.data_dir = Path(file)
        self.is_val = is_val
        self.clip_timepoints = clip_timepoints
        self.max_len = max_len
        self.norm = norm
        self.lm_name = lm_name
        self.GPT_training = GPT_training

        norm_file = self.data_dir / "normalization_params.npz"
        if norm_file.exists():
            params = np.load(norm_file)
            if self.norm == "robust":
                self.median = params["medians"]
                self.iqr = params["iqrs"] + 1e-8  # 增加平滑项
            elif self.norm == "std":
                self.mean = params["mean"]
                self.std = params["std"] + 1e-8   # 增加平滑项
        

        # 1. 加载 H5 数据 (基准：保留 2148 个数据)
        root_dir = "/root/MDD"
        h5_file_path = os.path.join(str(file), "fmri.h5") if os.path.isabs(str(file)) else os.path.join(root_dir, str(file), "fmri.h5")
        with h5py.File(h5_file_path, "r") as f:
            self.all_fmri_data = f["data"][:]
            
        # 2. 构建 ID->Label 映射字典
        excel_path = '/root/autodl-tmp/MDD/REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx'
        df_mdd = pd.read_excel(excel_path, sheet_name='MDD')
        df_hc = pd.read_excel(excel_path, sheet_name='Controls')
        label_map = {str(i).strip(): 1 for i in df_mdd['ID']}
        label_map.update({str(i).strip(): 0 for i in df_hc['ID']})

        # 3. 读取 CSV，严格按 H5 顺序对齐
        meta = pd.read_csv(self.data_dir / "dataset.csv")
        
        self.labels = []
        self.texts = []

        # 遍历 H5 的长度，确保数量完全一致
        for idx in range(len(self.all_fmri_data)):
            # 获取对应的 meta 行，如果 CSV 比 H5 短，则补空值
            if idx < len(meta):
                row = meta.iloc[idx]
                subj_id = str(row['subject_id']).strip()
                # 查找标签，找不到则标记为 2 (未标注)
                label = label_map.get(subj_id, 2)
                text = f"{row.get('fc_desc', '')} {row.get('gradient_desc', '')} {row.get('graph_desc', '')} {row.get('region_self_desc', '')}"
            else:
                # 如果 CSV 记录不足，填入默认值
                label = 2
                text = ""
            
            self.labels.append(label)
            self.texts.append(text)
        # # 在 __init__ 中 self.labels 赋值之后添加：
        #     print(f"DEBUG: 总数据量: {len(self.labels)}")
        #     print(f"DEBUG: 标签 unique 值: {np.unique(self.labels)}")
        #     print(f"DEBUG: 标签 0 的数量: {np.sum(np.array(self.labels) == 0)}")
        #     print(f"DEBUG: 标签 1 的数量: {np.sum(np.array(self.labels) == 1)}")
        # 4. 初始化 inds (放在所有数据列表准备好之后)
        if inds is None:
            self.inds = list(range(len(self.labels)))
        else:
            self.inds = inds

        self.num_samples = len(self.labels)
        
        # 2. 同样的，确保 inds 也在 __init__ 中定义
        if not hasattr(self, 'inds'):
            self.inds = list(range(self.num_samples))

        print(f"数据加载完成: H5={len(self.all_fmri_data)}, 标签={len(self.labels)}")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.prompts = [
            "Can you describe the functional characteristics of this brain?",
            "What does this fMRI scan show about the patient's brain activity?",
            "Analyze this fMRI data and provide a description.",
            "Based on the fMRI features, describe the brain state.",
            "Provide an overview of the neural patterns observed in this scan."
        ]

        # 在 __init__ 的最后
        self.num_samples = len(self.labels)
        # 强制让 inds 与 num_samples 一一对应
        self.inds = list(range(self.num_samples))

    def __len__(self):
        return self.num_samples

    def load_fmri(self, real_index):
        """
        从物理内存直接切片
        """
        X = self.all_fmri_data[real_index]
        X = torch.tensor(X, dtype=torch.float32)

        # 时间长度截断或填充至 clip_timepoints 维度
        if X.shape[1] > self.clip_timepoints:
            X = X[:, :self.clip_timepoints]
        elif X.shape[1] < self.clip_timepoints:
            X = X.unsqueeze(0)
            X = F.interpolate(X, size=self.clip_timepoints, mode='nearest').squeeze(0)

        # 内存数学计算
        if self.norm == "robust" and self.median is not None:
            X = (X - self.median) / (self.iqr + 1e-8)
        elif self.norm == "std" and self.mean is not None:
            X = (X - self.mean) / (self.std + 1e-8)

        return X

    def encode_text(self, text):
        """
        文本编码
        """
        prompt = random.choice(self.prompts)
        full_text = f"Question:\n{prompt}\n\nAnswer:\n{text}\n"

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return (
            encoding["input_ids"].squeeze(),
            encoding["attention_mask"].squeeze()
        )

    def __getitem__(self, index):
        """
        获取样本 (深度匹配主脚本多任务解包逻辑)
        """
        if index >= len(self.inds):
            print(f"DEBUG: 越界了! index={index}, len(inds)={len(self.inds)}")
            raise IndexError("索引越界")
            
        real_index = self.inds[index] # 这里的 index 是 DataLoader 给的
        
        # 确保 real_index 不会超出数据范围
        if real_index >= len(self.all_fmri_data):
            # 这种情况说明 inds 里的映射值有问题
            real_index = real_index % len(self.all_fmri_data)
        # 映射回全量预加载内存中的真实物理索引
        
        X = self.load_fmri(real_index)
        # 1. 提前获取标签，放在 return 之前
        label = self.labels[real_index]  # 确保这里用的是 real_index
        
        # # 2. 这里的调试代码只在第一次运行或抽样时有效，否则终端会爆掉
        # # 建议只在满足条件时打印一次，或者在调试时使用
        # if label == 1:
        #     print(f"DEBUG: 索引 {index} 对应标签为 1 (MDD), 物理索引: {real_index}")
        
        # 建立全1掩码张量
        fmri_gpt_mask = torch.ones(X.shape[0],X.shape[0], dtype=torch.long) 
        
        text = self.texts[real_index]
        text_input_ids, text_attention_mask = self.encode_text(text)

        # 根据当前是否是验证集，动态返回匹配主脚本解包数量的变量
        if self.is_val:
            # 验证阶段：主脚本期望解包 5 个值 (包含 reference_texts 原文)
            return (
                X,
                fmri_gpt_mask,
                text_input_ids,
                text_attention_mask,
                text,label
            )
        else:
            # 训练阶段：主脚本期望解包 4 个值
            return (
                X,
                fmri_gpt_mask,
                text_input_ids,
                text_attention_mask,label
            )
        


class fMRIDataSet(Dataset):
    def __init__(self, file, inds=None, norm='robust', GPT_training=False, patch_size=None, next_time_mask=False, clip_timepoints=160, **kwargs):
        base_dir = Path(file) 
        root_dir = "/root/MDD"
        if os.path.isabs(str(file)):
            h5_file_path = os.path.join(str(file), "fmri.h5")
        else:
            h5_file_path = os.path.join(root_dir, str(file), "fmri.h5")
             
        self.h5_file = h5_file_path
        print(f"DEBUG: 尝试访问 H5 路径: {h5_file_path}")

        if not os.path.exists(h5_file_path):
            print(f"错误：找不到文件 {h5_file_path}")
            print(f"当前工作目录: {os.getcwd()}")
            raise FileNotFoundError(f"无法找到 {h5_file_path}")

        self.h5_file = str(h5_file_path)
        print(f"DEBUG: 最终确定的 H5 路径是: {self.h5_file}")

        with h5py.File(self.h5_file, 'r') as file_handle:
            self.keys = [str(i) for i in range(len(file_handle['data']))]
            self.subjs = self.keys
            self.sess = self.keys

        self.inds = inds if inds is not None else list(range(len(self.keys)))
        self.GPT_training = GPT_training
        self.next_time_mask = next_time_mask  
        self.norm = norm
        self.patch_size = patch_size
        self.clip_timepoints = clip_timepoints

        norm_params_path = base_dir / 'normalization_params.npz'
        if not norm_params_path.exists():
            print(f"CRITICAL ERROR: 归一化参数文件不存在! 路径: {norm_params_path}")
            raise FileNotFoundError(f"无法找到归一化参数文件: {norm_params_path}")
        
        norm_params = np.load(norm_params_path)
        if norm == 'robust':
            self.median, self.iqr = norm_params['medians'], norm_params['iqrs']
        elif norm == 'std':
            self.mean, self.std = norm_params['mean'], norm_params['std']

    def __len__(self):
        return len(self.inds)


def get_fmri_data(file, data_cls=fMRIDataSet, train_ratio=1, val_ratio=0.2, **kwargs):
    """
    建立训练集与验证集的数据分配调度中心
    """
    files = file if isinstance(file, list) else [file]
    
    # 建立临时数据集以获取样本总量统计
    temp_dataset = fMRIDataSet(files[0], **kwargs)
    total_samples = len(temp_dataset)
    
    # 划分训练集与验证集索引 (直接按照主脚本中传入的 ratio 生成不重叠索引)
    if train_ratio == 1:
        train_inds = list(range(total_samples))
        val_inds = np.random.choice(total_samples, size=int(total_samples * val_ratio), replace=False).tolist()
    else:
        split_point = int(total_samples * train_ratio)
        train_inds = list(range(split_point))
        val_inds = list(range(split_point, total_samples))
    
    # 实例化真正的 Dataset 类
    train_sets = [data_cls(f, inds=train_inds, is_val=False, **kwargs) for f in files]
    val_sets = [data_cls(f, inds=val_inds, is_val=True, **kwargs) for f in files]
    
    train_set = ConcatDataset(train_sets) if len(files) > 1 else train_sets[0]
    val_set = ConcatDataset(val_sets) if len(files) > 1 else val_sets[0]
    
    return train_set, val_set

def get_data_info(target_name):
    """
    根据任务标签自动返回其评估属性。
    根据 train_instruction.py 的逻辑：
    num_classes >= 2 为分类任务（计算 ACC / AUC）；num_classes = 1 为回归任务（计算 MAE / R2）。
    """
    # 假设你的抑郁症分类标签叫 'MDD' 或 'label' 或 'status'
    if target_name.lower() in ['mdd', 'label', 'status', 'group']:
        return {'num_classes': 2, 'type': 'classification'}
    # 如果有年龄等连续回归变量
    elif target_name.lower() in ['age', 'score']:
        return {'num_classes': 1, 'type': 'regression'}
    else:
        # 默认作为二分类处理
        return {'num_classes': 2, 'type': 'classification'}

def select_few_shot_indices(dataset, train_inds, fewshot_samples):
    """
    从训练集索引中，为 MDD 分类任务平衡地抽取少量样本（Few-shot）。
    确保抽出来的少样本里，健康对照（HC）和患者（MDD）的比例尽量是 1:1。
    """
    import random
    # 1. 收集所有训练样本的标签
    labels = []
    for idx in train_inds:
        # 假设你的数据集对象可以通过类似 dataset.labels 或自定义属性获取单个标签
        # 如果你的 Dataset 实现了通过索引取标签，可以这样获取；否则需要根据你的类设计做微调
        try:
            # 尝试通过基础 dataset 的内部结构取标签（这里假设 __getitem__ 返回的字典有 label 或 y）
            sample = dataset[idx]
            label = sample.get('label', sample.get('y', 0))
            if isinstance(label, torch.Tensor):
                label = label.item()
            labels.append(label)
        except Exception:
            labels.append(0) # 兜底

    labels = np.array(labels)
    train_inds = np.array(train_inds)

    # 2. 按类别（0和1）分别提取索引
    unique_labels = np.unique(labels)
    samples_per_class = fewshot_samples // len(unique_labels) if len(unique_labels) > 0 else fewshot_samples
    
    selected_inds = []
    for cls in unique_labels:
        cls_mask = (labels == cls)
        cls_inds = train_inds[cls_mask]
        
        if len(cls_inds) >= samples_per_class:
            # 随机抽取该类别的样本
            chosen = np.random.choice(cls_inds, samples_per_class, replace=False)
        else:
            # 如果样本不够，有多少取多少
            chosen = cls_inds
        selected_inds.extend(chosen.tolist())
        
    # 如果总数因为整除或样本不够没凑够，用原样本补齐
    if len(selected_inds) < fewshot_samples and len(train_inds) > 0:
        remaining = list(set(train_inds) - set(selected_inds))
        needed = fewshot_samples - len(selected_inds)
        if len(remaining) >= needed:
            selected_inds.extend(np.random.choice(remaining, needed, replace=False).tolist())
        else:
            selected_inds.extend(remaining)

    random.shuffle(selected_inds)
    return selected_inds

# ==========================================
# 2. 核心适配接口：供 train_instruction.py 调用
# ==========================================
def get_fmri_data_inst(
    batch_size, 
    val_batch_size, 
    datasets=['MDD'],  
    train_val_test_ratio=[0.7, 0.1, 0.2], 
    dataset_target_mapping=None, 
    dataset_config_dict=None, 
    separate_multi_task_loaders=False, 
    fewshot_samples=0, 
    **kwargs
):
    """
    获取适配 MDD 数据的 fMRI 指令微调数据集和加载器。
    已硬编码锁定路径与数据集名称，彻底杜绝外层参数污染。
    """
    # 1. 强行指定正确的类名
    dataset_cls = fMRITextDataset
    
    # 2. 核心拦截：无论外层传进来什么，强行重写为 MDD，防止 UKB 报错
    datasets = ['MDD']
    all_target_names = {'MDD': ['MDD']}
    
    train_sets = []
    train_loaders_dict = {}  
    val_test_loaders = {}

    for name in datasets: # 此时 name 必定为 'MDD'
        dataset_config = dataset_config_dict.get(name, {}) if dataset_config_dict else {}
        is_multi = dataset_config.get('is_multi', False)
        
        if is_multi:
            raise NotImplementedError("MDD 数据集目前仅支持单任务分类/回归。")
        else:
            targets = all_target_names.get(name, ['MDD'])
            for target_name in targets: # 此时 target_name 必定为 'MDD'
                
                # 3. 实例化完整数据集（强行指定绝对路径 /root/autodl-tmp/MDD）
                dataset = dataset_cls(file='/root/autodl-tmp/MDD', dataset_name='MDD', target_name=target_name, **kwargs)
                num_samples = len(dataset)

                # 4. 强行指定索引文件的物理路径
                inds_path = f'/root/autodl-tmp/MDD/label_inds_{target_name}.npy'
                if os.path.exists(inds_path):
                    label_inds = np.load(inds_path, allow_pickle=True).item()
                    train_inds = label_inds['train_inds']
                    val_inds = label_inds['val_inds']
                    test_inds = label_inds['test_inds']
                else:
                    # 如果不存在切分好的索引文件，则根据比例硬切分
                    indices = list(range(num_samples))
                    r_train, r_val, _ = train_val_test_ratio
                    
                    split1 = int(num_samples * r_train)
                    split2 = int(num_samples * (r_train + r_val))
                    
                    train_inds = indices[:split1]
                    val_inds = indices[split1:split2]
                    test_inds = indices[split2:]

                # 5. 处理 Few-shot（少样本微调）逻辑
                if fewshot_samples > 0:
                    train_inds = select_few_shot_indices(dataset, train_inds, fewshot_samples)
                    val_batch_size = 64

                # 6. 构建各自的子数据集（同样强行指定绝对路径）
                train_set = dataset_cls(file='/root/autodl-tmp/MDD', dataset_name='MDD', inds=train_inds, target_name=target_name, is_val=False, **kwargs)
                val_set = dataset_cls(file='/root/autodl-tmp/MDD', dataset_name='MDD', inds=val_inds, target_name=target_name, is_val=True, **kwargs)
                test_set = dataset_cls(file='/root/autodl-tmp/MDD', dataset_name='MDD', inds=test_inds, target_name=target_name, is_val=True, **kwargs)

                # 7. 强行指定字典键名，防止主脚本后续读取时报 KeyError
                dataset_key = 'MDD-MDD'
                
                # 8. 构建 PyTorch DataLoader
                if separate_multi_task_loaders:
                    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
                    train_loaders_dict[dataset_key] = train_loader
                else:
                    train_sets.append(train_set)

                val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
                test_loader = DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)

                # 获取该预测任务的类型与类别数信息
                data_info = get_data_info(target_name)
                val_test_loaders[dataset_key] = {'val': val_loader, 'test': test_loader, 'info': data_info}

    # 9. 统一返回
    if separate_multi_task_loaders:
        return train_loaders_dict, val_test_loaders
    else:
        if len(train_sets) == 0:
            raise ValueError("没有成功加载任何 MDD 训练子集，请检查数据集名称及路径。")
        train_set_merge = ConcatDataset(train_sets)
        train_loader = DataLoader(train_set_merge, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
        return train_loader, val_test_loaders
