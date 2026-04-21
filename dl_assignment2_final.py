"""
Digit Detection using Faster R-CNN (ResNet50-V2)
This script trains an object detection model to identify digits in complex images.
It utilizes Automatic Mixed Precision (AMP) and AdamW for efficient training.
"""

import os
import json
import zipfile
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image

# ==========================================
# Configuration & Paths
# ==========================================
# Standard local repository structure assumes data is in a './dataset' folder
DATA_DIR = './dataset'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test')
TRAIN_JSON = os.path.join(DATA_DIR, 'train.json')
OUTPUT_DIR = './output'

NUM_EPOCHS = 6
NUM_CLASSES = 11
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# ==========================================
# Dataset Definitions
# ==========================================
def get_transform():
    """Returns the standard tensor transformation."""
    return T.Compose([T.ToTensor()])

class TrainDigitDataset(Dataset):
    """
    Custom Dataset class for parsing COCO-format training annotations 
    and loading digit images.
    """
    def __init__(self, img_dir, annotation_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
            
        self.img_records = {img['id']: img for img in self.coco_data['images']}
        self.img_ids = list(self.img_records.keys())
        self.ann_records = {img_id: [] for img_id in self.img_ids}
        
        for ann in self.coco_data['annotations']:
            self.ann_records[ann['image_id']].append(ann)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.img_records[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
            
        anns = self.ann_records[img_id]
        boxes, labels = [], []
        
        for ann in anns:
            x_min, y_min, width, height = ann['bbox']
            boxes.append([x_min, y_min, x_min + width, y_min + height])
            labels.append(ann['category_id'])
            
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}
        return img, target

    def __len__(self):
        return len(self.img_ids)

class TestDigitDataset(Dataset):
    """Dataset class for loading test images for inference."""
    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(img_dir)))
        
    def __getitem__(self, idx):
        import re
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
            
        match = re.search(r'\d+', img_name)
        img_id = int(match.group()) if match else idx
        return img, img_id

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    """Custom collate function for bounding box batches."""
    return tuple(zip(*batch))

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == '__main__':
    # Disable cudnn benchmark to prevent recompilation overhead for dynamic image sizes
    torch.backends.cudnn.benchmark = False 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Initializing training on device: {device}")

    if not os.path.exists(TRAIN_JSON):
        raise FileNotFoundError(f"Dataset not found at {DATA_DIR}. Please extract files.")

    # 1. Model Initialization
    # Complies with assignment constraint: Backbone pre-trained only.
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None, 
        weights_backbone=ResNet50_Weights.DEFAULT, 
        num_classes=NUM_CLASSES,
        min_size=800,  
        max_size=1333   
    )
    model = model.to(device)

    # 2. Data Loaders
    train_dataset = TrainDigitDataset(TRAIN_IMG_DIR, TRAIN_JSON, transforms=get_transform())
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=2, 
        pin_memory=True,
        persistent_workers=True 
    )

    # 3. Optimizer and Scheduler Setup
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    # 4. Training Loop
    print(f"Starting Mixed Precision Training for {NUM_EPOCHS} Epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device, non_blocking=True) for image in images)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += losses.item()
            
            if (i + 1) % 500 == 0:
                print(f"  Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {losses.item():.4f}")
                
        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        current_lr = lr_scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    # 5. Inference and Submission Generation
    print("Training complete. Starting inference on test dataset...")
    test_dataset = TestDigitDataset(TEST_IMG_DIR, transforms=get_transform())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model.eval()
    predictions_list = []

    with torch.no_grad():
        for images, image_ids in test_loader:
            images = list(img.to(device, non_blocking=True) for img in images)
            outputs = model(images)
            
            image_id = int(image_ids[0].item())
            boxes = outputs[0]['boxes'].cpu().numpy()
            scores = outputs[0]['scores'].cpu().numpy()
            labels = outputs[0]['labels'].cpu().numpy()
            
            # Post-processing filter optimized for Recall
            for box, score, label in zip(boxes, scores, labels):
                if score > 0.05: 
                    x_min, y_min = float(box[0]), float(box[1])
                    width, height = float(box[2] - box[0]), float(box[3] - box[1])
                    
                    predictions_list.append({
                        "image_id": image_id,
                        "bbox": [x_min, y_min, width, height],
                        "score": float(score),
                        "category_id": int(label)
                    })

    # 6. Save Outputs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, 'pred.json')
    
    with open(json_path, 'w') as f:
        json.dump(predictions_list, f, indent=4)

    zip_path = os.path.join(OUTPUT_DIR, 'submission.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, arcname='pred.json')

    print(f"Inference complete. Output saved to {zip_path}")