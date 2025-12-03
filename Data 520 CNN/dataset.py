# Custom Dataset class
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RiceDataset(Dataset):
    def __init__(self, csv_file, s1_dir, s2_dir, transform=None):
        """
        Args:
            csv_file (string): CSV 文件的路径.
            s1_dir (string): Sentinel-1 图片文件夹路径.
            s2_dir (string): Sentinel-2 图片文件夹路径.
            transform (callable, optional): 数据增强函数.
        """
        self.df = pd.read_csv(csv_file)
        self.s1_dir = s1_dir
        self.s2_dir = s2_dir
        self.transform = transform
        
        self.default_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        

        if 'is_rice' in self.df.columns:
            label = int(row['is_rice'])
        elif 'label' in self.df.columns:
            label = int(row['label'])
        else:
            label = 0 


        year = int(row['year']) if 'year' in self.df.columns else 2019
        

        filename_v1 = f"{label}_{year}_{idx}.jpg"  
        filename_v2 = f"{label}_{idx}.jpg"       
        

        s1_path = self._find_path(self.s1_dir, filename_v1, filename_v2)
        if s1_path:
            img_s1 = Image.open(s1_path).convert('RGB')
        else:
       
            img_s1 = Image.new('RGB', (64, 64), (0, 0, 0))


        s2_path = self._find_path(self.s2_dir, filename_v1, filename_v2)
        if s2_path:
            img_s2 = Image.open(s2_path).convert('RGB')
        else:
            img_s2 = Image.new('RGB', (64, 64), (0, 0, 0))


        seed = torch.randint(0, 2**32, (1,)).item()
        
        if self.transform:
            torch.manual_seed(seed)
            img_s1 = self.transform(img_s1)
            
            torch.manual_seed(seed)
            img_s2 = self.transform(img_s2)
        else:
            img_s1 = self.default_transform(img_s1)
            img_s2 = self.default_transform(img_s2)

        return {
            's1': img_s1,
            's2': img_s2,
            'label': torch.tensor(label, dtype=torch.long)
        }

    def _find_path(self, dir_path, name1, name2):
        """辅助函数：尝试找名字1，找不到就找名字2"""
        p1 = os.path.join(dir_path, name1)
        if os.path.exists(p1): return p1
        
        p2 = os.path.join(dir_path, name2)
        if os.path.exists(p2): return p2
        
        return None