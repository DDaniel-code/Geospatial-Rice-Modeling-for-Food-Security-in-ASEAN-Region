import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

# 🔥 必须从 model_frozen 导入模型
from dataset import RiceDataset
from model_frozen import DualStreamRiceModel 

def train_model():
    # --- 1. 配置参数 ---
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3  # 冻结模式下，可以用标准学习率
    EPOCHS = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Device: {DEVICE}")

    # --- 2. 路径设置 ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 指向我们刚刚修复好的 CSV
    csv_file = os.path.join(project_root, 'data/origin/balanced_data_fixed.csv')
    s1_dir = os.path.join(project_root, 'data/raw_images/s1_radar')
    s2_dir = os.path.join(project_root, 'data/raw_images/s2_optical')
    
    if not os.path.exists(csv_file):
        print(f"❌ 找不到数据文件: {csv_file}")
        print("请先运行 src/fix_and_balance.py 生成数据！")
        return

    # --- 3. 数据加载 ---
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = RiceDataset(csv_file, s1_dir, s2_dir, transform=data_transforms)
    
    # 8:2 切分
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"📊 Data Loaded: Train={len(train_dataset)} | Val={len(val_dataset)}")

    # --- 4. 初始化模型 (开启冻结) ---
    model = DualStreamRiceModel(num_classes=2, freeze_backbone=True).to(DEVICE)
    
    # 优化器：只更新那些 requires_grad=True 的层 (也就是最后那几层)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # --- 5. 训练循环 ---
    best_acc = 0.0
    save_dir = os.path.join(project_root, 'results/checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # 进度条 (在 Longleaf 日志里可能显示不全，但这不影响运行)
            for batch in tqdm(dataloader, desc=phase, ncols=80, leave=False):
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

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_frozen.pth'))
                print(f"🔥 Best Acc: {best_acc:.4f}")

    print(f'Done. Best Val Acc: {best_acc:.4f}')

if __name__ == '__main__':
    train_model()