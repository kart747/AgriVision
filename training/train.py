"""
AgriVision AI - EfficientNetB0 Training Script (Improved)
Optimized for higher F1 score (~0.90 target)

Usage:
  python train.py           # Full training
  python train.py --eval   # Evaluation only
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

EVAL_ONLY = "--eval" in sys.argv

# CONFIG
DATASET_PATH = "/home/pranam/Downloads/plantvillage-dataset/plantvillage dataset/color"
WEIGHTS_DIR  = "/home/pranam/Downloads/AgriVision/backend/model/weights"
BATCH_SIZE   = 32
IMG_SIZE     = 224
FROZEN_EPOCHS   = 5
UNFROZEN_EPOCHS = 40  # Increased from 30
WARMUP_EPOCHS    = 5
LABEL_SMOOTHING  = 0.1
MIXUP_ALPHA      = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable FP16 - disabled for stability
USE_AMP = False

TARGET_CLASSES = [
    "Tomato___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Target_Spot", "Tomato___Spider_mites Two-spotted_spider_mite", 
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Apple___healthy", "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Grape___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"
]

os.makedirs(WEIGHTS_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if USE_AMP:
        print("FP16 Mixed Precision: Enabled ✓")

# TRANSFORMS - Enhanced augmentation
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# FILTER DATASET
full_dataset = datasets.ImageFolder(DATASET_PATH)
target_class_set = set(TARGET_CLASSES)
target_class_names = [c for c in full_dataset.classes if c in target_class_set]
class_to_idx = {c: i for i, c in enumerate(target_class_names)}

filtered_indices = [i for i in range(len(full_dataset)) if full_dataset.classes[full_dataset.targets[i]] in target_class_set]

class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, full_dataset, filtered_indices, class_to_idx):
        self.full_dataset = full_dataset
        self.filtered_indices = filtered_indices
        self.class_to_idx = class_to_idx
    def __len__(self):
        return len(self.filtered_indices)
    def __getitem__(self, idx):
        img, label = self.full_dataset[self.filtered_indices[idx]]
        new_label = self.class_to_idx[self.full_dataset.classes[label]]
        return img, new_label

filtered_dataset = FilteredDataset(full_dataset, filtered_indices, class_to_idx)
print(f"Total filtered images: {len(filtered_dataset)}")
print(f"Classes: {len(target_class_names)}")

# SAVE CLASS NAMES
with open(os.path.join(WEIGHTS_DIR, "class_names.json"), "w") as f:
    json.dump(target_class_names, f, indent=2)

# SPLIT
n_total = len(filtered_dataset)
n_train, n_val = int(0.8 * n_total), int(0.1 * n_total)
n_test = n_total - n_train - n_val
train_ds, val_ds, test_ds = random_split(filtered_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset, self.transform = subset, transform
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img) if self.transform else img, label
    def __len__(self): return len(self.subset)

train_loader = DataLoader(TransformDataset(train_ds, train_transforms), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(TransformDataset(val_ds, val_transforms),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(TransformDataset(test_ds, val_transforms),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# CLASS WEIGHTS
train_labels = [train_ds[i][1] for i in range(len(train_ds))]
class_weights = compute_class_weight('balanced', classes=np.arange(len(target_class_names)), y=train_labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# MODEL
num_classes = len(target_class_names)
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
for p in model.parameters(): p.requires_grad = False
model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, num_classes))
model = model.to(DEVICE)

# Label smoothing loss
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING)

# MixUp function
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Mixed precision scaler
scaler = GradScaler('cuda') if USE_AMP else None

def run_epoch(loader, training=True, optimizer=None, use_mixup=True):
    model.train() if training else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    
    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            if training and use_mixup and np.random.random() < 0.5:
                imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)
                
                if USE_AMP:
                    with autocast():
                        outputs = model(imgs)
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    outputs = model(imgs)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                
                if training:
                    optimizer.zero_grad()
                    if USE_AMP:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
            else:
                if USE_AMP:
                    with autocast():
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                
                if training:
                    optimizer.zero_grad()
                    if USE_AMP:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                    optimizer.step()
            
            total_loss += loss.item() * imgs.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader.dataset), f1_score(all_labels, all_preds, average='macro', zero_division=0)

print(f"Model ready. Training on {num_classes} classes.")
print(f"Improvements: MixUp (α={MIXUP_ALPHA}), Label Smoothing ({LABEL_SMOOTHING}), FP16={USE_AMP}")

# TRAINING (only if not eval mode)
if EVAL_ONLY:
    print("\n" + "="*50)
    print("EVALUATION ONLY MODE")
    print("="*50)
    if not os.path.exists(os.path.join(WEIGHTS_DIR, "best_model.pth")):
        print("ERROR: No best_model.pth found. Run training first.")
        sys.exit(1)
    checkpoint = torch.load(os.path.join(WEIGHTS_DIR, "best_model.pth"), map_location=DEVICE)
    model.load_state_dict(checkpoint)
    print("Loaded best_model.pth")
else:
    # Backup existing model
    if os.path.exists(os.path.join(WEIGHTS_DIR, "best_model.pth")):
        backup_path = os.path.join(WEIGHTS_DIR, "best_model_backup.pth")
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy(os.path.join(WEIGHTS_DIR, "best_model.pth"), backup_path)
            print(f"Backed up existing model to best_model_backup.pth")
    
    print("\n" + "="*50)
    print("PHASE 1: Training classifier head (frozen)")
    print("="*50)
    best_f1 = 0.0
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=FROZEN_EPOCHS)
    
    for epoch in range(1, FROZEN_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_f1 = run_epoch(train_loader, True, optimizer, use_mixup=False)
        val_loss,   val_f1   = run_epoch(val_loader, False)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{FROZEN_EPOCHS} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | {elapsed:.0f}s")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best_model.pth"))
            print(f"  -> Saved best_model.pth (F1: {best_f1:.4f})")

    print("\n" + "="*50)
    print("PHASE 2: Fine-tuning (unfrozen with warmup)")
    print("="*50)
    for p in model.parameters(): p.requires_grad = True
    
    # Lower learning rate for fine-tuning
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    
    # Warmup + Cosine annealing
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cos_scheduler = CosineAnnealingLR(optimizer, T_max=UNFROZEN_EPOCHS - WARMUP_EPOCHS)
    
    for epoch in range(1, UNFROZEN_EPOCHS + 1):
        t0 = time.time()
        
        # Use mixup after warmup
        use_mixup = epoch > WARMUP_EPOCHS
        
        train_loss, train_f1 = run_epoch(train_loader, True, optimizer, use_mixup=use_mixup)
        val_loss,   val_f1   = run_epoch(val_loader, False)
        
        # Update schedulers
        if epoch <= WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            cos_scheduler.step()
        
        elapsed = time.time() - t0
        print(f"Epoch {FROZEN_EPOCHS+epoch}/{FROZEN_EPOCHS+UNFROZEN_EPOCHS} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | {elapsed:.0f}s")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best_model.pth"))
            print(f"  -> Saved best_model.pth (F1: {best_f1:.4f})")
    
    print(f"\nBest Validation F1: {best_f1:.4f}")

# Test Eval
print("\n" + "="*50)
print("FINAL TEST EVALUATION")
print("="*50)
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        all_preds.extend(model(imgs.to(DEVICE)).argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())

final_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
print(f"\nFinal Test F1: {final_f1:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=target_class_names, zero_division=0))

print("\n" + "="*50)
print("COMPLETE!")
print("="*50)
print(f"Weights: {WEIGHTS_DIR}/best_model.pth")
print(f"Classes: {WEIGHTS_DIR}/class_names.json")
