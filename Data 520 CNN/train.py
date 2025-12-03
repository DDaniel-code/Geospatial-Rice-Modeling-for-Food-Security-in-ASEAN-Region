import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

# 引入我们刚才定义的 Dataset 和 Model
#from dataset import RiceDataset
from model import DualStreamRiceModel

def train_model():
    # ================= 配置参数 =================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 15
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Training Device: {DEVICE}")

    # ================= 1. 动态设置路径 =================
    # 获取 src 文件夹的上一级目录 (即 520 项目根目录)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 拼接数据路径
    csv_file = os.path.join(project_root, 'data/origin/thailand_riceland_gee_data-6.csv')
    s1_dir = os.path.join(project_root, 'data/raw_images/s1_radar')
    s2_dir = os.path.join(project_root, 'data/raw_images/s2_optical')
    
    # 检查路径是否存在
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_file}")
    if not os.path.exists(s1_dir):
        raise FileNotFoundError(f"找不到 S1 图片文件夹: {s1_dir}")
        
    print(f"📂 Project Root: {project_root}")
    print(f"📂 CSV Path: {csv_file}")

    # ================= 2. 数据准备 =================
    # 定义增强策略 (旋转、翻转、归一化)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 初始化数据集
    full_dataset = RiceDataset(csv_file, s1_dir, s2_dir, transform=data_transforms)
    
    # 划分训练集 (80%) 和验证集 (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"📊 Dataset Loaded: Train={len(train_dataset)} | Val={len(val_dataset)}")

    # ================= 3. 初始化模型 =================
    # DualStreamRiceModel 来自 src/model.py
    model = DualStreamRiceModel(num_classes=2).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # ================= 4. 训练循环 =================
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 结果保存路径
    save_dir = os.path.join(project_root, 'results/checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # 遍历批次
            for batch in tqdm(dataloader, desc=phase):
                s1 = batch['s1'].to(DEVICE)
                s2 = batch['s2'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(s1, s2)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * s1.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                print(f"🔥 New Best Accuracy: {best_acc:.4f}")

    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')

if __name__ == '__main__':
    train_model()