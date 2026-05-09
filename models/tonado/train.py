import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

from dataset import TomatoDataset, get_train_transform, get_valid_transform

# --- CẤU HÌNH THAM SỐ ---
DATA_ROOT = r"E:\nckh_cachua\data"
CHECKPOINT_DIR = r"E:\nckh_cachua\checkpoints"
LOG_DIR = r"E:\nckh_cachua\logs_tomato_final"
NUM_CLASSES = 9  # 8 lớp bệnh + 1 background
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.0003
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_model(num_classes):
    # Sử dụng Faster R-CNN với backbone MobileNetV3
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train():
    # 1. Dataset & DataLoader
    train_dataset = TomatoDataset(DATA_ROOT, 'train', get_train_transform())
    valid_dataset = TomatoDataset(DATA_ROOT, 'valid', get_valid_transform())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_fn, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              collate_fn=collate_fn, num_workers=2)

    # 2. Model, Optimizer, Scheduler
    model = get_model(NUM_CLASSES).to(DEVICE)
    
    # Chiến lược: Đóng băng backbone ở 2 Epoch đầu để ổn định đầu ra
    # Sau đó sẽ unfreeze ở code huấn luyện 
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.0005)
    
    # Scheduler hình Cosine giúp hội tụ mượt mà
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 3. Logging & TensorBoard
    writer = SummaryWriter(LOG_DIR)
    
    best_loss = float('inf')
    print(f" Bắt đầu huấn luyện trên thiết bị: {DEVICE}")
    print(f" Theo dõi tại TensorBoard: tensorboard --logdir={LOG_DIR}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        loss_classifier_accum = 0
        loss_box_reg_accum = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for images, targets in pbar:
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            loss_classifier_accum += loss_dict['loss_classifier'].item()
            loss_box_reg_accum += loss_dict['loss_box_reg'].item()
            
            pbar.set_postfix(loss=losses.item(), cls=loss_dict['loss_classifier'].item(), box=loss_dict['loss_box_reg'].item())

        avg_loss = epoch_loss / len(train_loader)
        lr_scheduler.step()

        # Log TensorBoard chi tiết
        writer.add_scalar("Loss/Total", avg_loss, epoch)
        writer.add_scalar("Loss/Classifier", loss_classifier_accum / len(train_loader), epoch)
        writer.add_scalar("Loss/Box_Reg", loss_box_reg_accum / len(train_loader), epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

        # 4. Lưu Checkpoint (Lưu cả trạng thái optimizer để có thể train tiếp)
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f" Đã lưu mô hình tốt nhất (Loss: {avg_loss:.4f})")

        # Luôn lưu mô hình mới nhất
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last_model.pth"))

    writer.close()
    print("Huấn luyện hoàn tất! Mô hình tốt nhất tại: " + os.path.join(CHECKPOINT_DIR, "best_model.pth"))

if __name__ == "__main__":
    train()
