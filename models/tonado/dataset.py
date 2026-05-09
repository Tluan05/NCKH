import os
import torch
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Danh sách 8 lớp bệnh cà chua 
CLASSES = [
    'background', 'Healthy_Tomato', 'Blossom_End_Rot', 'Late_Blight', 
    'Mold', 'Anthracnose', 'Fruit_Cracking', 'Catfaced', 'Spotted_Wilt_Virus'
]

class TomatoDataset(Dataset):
    def __init__(self, root_dir, split='train', transforms=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(self.root_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.class_to_idx = {name: i for i, name in enumerate(CLASSES)}

    def __getitem__(self, idx):
        # 1. Đọc ảnh
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0 # Chuẩn hóa về [0, 1]

        # 2. Đọc file nhãn XML Pascal VOC
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        xml_path = os.path.join(self.root_dir, xml_name)
        
        boxes = []
        labels = []
        areas = []
        
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name not in self.class_to_idx:
                    continue
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_to_idx[name])
                areas.append((xmax - xmin) * (ymax - ymin))

        # Chuyển sang tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # 3. Áp dụng các phép biến đổi (Augmentation)
        if self.transforms is not None:
            sample = self.transforms(image=image, bboxes=target["boxes"], labels=labels.tolist())
            image = sample['image']
            target['boxes'] = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(sample['labels'], dtype=torch.int64)
            
            if len(target['boxes']) == 0:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['labels'] = torch.zeros((0,), dtype=torch.int64)

        return image, target

    def __len__(self):
        return len(self.image_files)

# Định nghĩa các phép biến đổi ảnh nâng cao
def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),
        A.Resize(640, 640),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_valid_transform():
    return A.Compose([
        A.Resize(640, 640),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

