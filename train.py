# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np
import multiprocessing


def main():
    multiprocessing.freeze_support()

    # ===== 新增：详细的 GPU 检测报告 =====
    print("============== GPU 检测报告 ==============")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ 未检测到 GPU，将使用 CPU 训练（速度较慢）")
    print("========================================\n")

    # ===== 保持原有配置 =====
    DATA_DIR = "data"
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE} | PyTorch version: {torch.__version__}")
    # =============== 数据预处理（适配你的数据结构）===============
    # 训练时增强：随机翻转+裁剪
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证时：只做标准化
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ===============  关键修改：加载你的数据结构 ===============
    # 注意：ImageFolder 要求子文件夹名就是类别名
    full_dataset = datasets.ImageFolder(
        root=DATA_DIR,  # 根目录是 "data"
        transform=None  # 先不应用 transform，后面手动处理
    )

    # 检查类别映射（确保 cats=0, dogs=1）
    print(" Dataset classes:", full_dataset.classes)  # 应该输出 ['cats', 'dogs']
    print(" Total images:", len(full_dataset))

    # ===============  关键修改：80/20 自动划分 ===============
    # 固定随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 获取所有索引并打乱
    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)

    # 80% 训练, 20% 验证
    val_size = int(0.2 * len(full_dataset))
    train_indices, val_indices = indices[val_size:], indices[:val_size]

    # 创建自定义数据集应用不同 transform
    class CustomSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform=None):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            if self.transform:
                img = self.transform(img)
            return img, label

    # 应用不同 transform
    train_dataset = CustomSubset(full_dataset, train_indices, transform_train)
    val_dataset = CustomSubset(full_dataset, val_indices, transform_val)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 修复：Windows 默认设为0，避免多进程问题
        pin_memory=True  # GPU 加速关键
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # 修复：Windows 默认设为0
        pin_memory=True
    )

    print(f" Dataset split: {len(train_dataset)} train | {len(val_dataset)} val")
    print(f" Model will use: {DEVICE}")

    # =============== 模型定义（ResNet18 适配 PyTorch 2.5.1）===============
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 自动下载预训练权重
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # 防止过拟合
        nn.Linear(num_ftrs, 2)  # 2 classes: cats(0), dogs(1)
    )
    model = model.to(DEVICE)

    # =============== 训练配置 ===============
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        verbose=True  # 修复：移除警告（已处理）
    )

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    # =============== 训练循环 ===============
    for epoch in range(NUM_EPOCHS):
        # ----- 训练阶段 -----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100. * train_correct / train_total:.1f}%"
            })

        train_acc = 100. * train_correct / train_total

        # ----- 验证阶段 -----
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="  Validation", leave=False):
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # 更新学习率（使用新API）
        scheduler.step(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best.pth")
            print(f"🏆 New best model saved! Val Acc: {val_acc:.2f}%")

        print(f" Epoch {epoch + 1} | "
              f"Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")

    print(f"\n Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f" Final model saved to: checkpoints/best.pth")


if __name__ == '__main__':
    main()  # 仅主进程执行