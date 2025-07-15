vlmeval/datasets/

from vlmeval.datasets.base import BaseDataset
from vlmeval.utils import split_dict

class MIRB(BaseDataset):
    def __init__(self, dataset="MIRB", **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset
        
        # MIRB 的特殊初始化参数
        self.mirb_version = kwargs.get('mirb_version', 'v1.0')
        self.caption_type = kwargs.get('caption_type', 'detailed')
        
        # 根据版本加载不同数据集
        if dataset == "MIRB_MINI":
            self.load_mini_dataset()
        elif dataset == "MIRB_FULL":
            self.load_full_dataset()

    def load_mini_dataset(self):
        # 从 InternVL 迁移的数据加载逻辑
        from internvl_chat.data import load_mirb_mini
        self.data = load_mirb_mini(self.mirb_version, self.caption_type)
        
        # 分割训练/测试集 (7:3)
        train_idx, test_idx = split_dict(self.data, 0.7)
        self.train = {k: self.data[k] for k in train_idx}
        self.test = {k: self.data[k] for k in test_idx}

    def load_full_dataset(self):
        # 完整数据集加载逻辑
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "image_path": item['image_path'],
            "question": item['question'],
            "gt_answer": item['answer'],
            "options": item.get('options', []),
            "task_type": item['task_type']
        }



