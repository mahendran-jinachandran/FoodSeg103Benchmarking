
import torch
import time
import os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
from torch import optim
import albumentations as A
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms as T
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

class CityscapesDataset(Dataset):
    def __init__(self, image_dir, label_dir, split='train', transform=None, target_transform=None, max_samples=None):
        self.image_dir = os.path.join(image_dir, split)
        self.label_dir = os.path.join(label_dir, split)
        self.transform = transform
        self.target_transform = target_transform
        self.image_paths, self.label_paths = [], []

        for city in os.listdir(self.image_dir):
            img_folder = os.path.join(self.image_dir, city)
            label_folder = os.path.join(self.label_dir, city)
            for file in os.listdir(img_folder):
                if file.endswith('_leftImg8bit.png'):
                    image_path = os.path.join(img_folder, file)
                    label_file = file.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
                    label_path = os.path.join(label_folder, label_file)
                    if os.path.exists(label_path):
                        self.image_paths.append(image_path)
                        self.label_paths.append(label_path)
                        if max_samples and len(self.image_paths) >= max_samples:
                            break
            if max_samples and len(self.image_paths) >= max_samples:
                break

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx])
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label.squeeze(0) if label.dim() == 3 else label


    
def compute_iou(preds, labels, num_classes=19, ignore_index=255):
    iou_metric = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=ignore_index).to('cpu')

    preds = preds.detach().cpu().flatten()
    labels = labels.detach().cpu().flatten()

    # Remove ignore_index pixels
    mask = labels != ignore_index
    preds = preds[mask]
    labels = labels[mask]

    preds = torch.clamp(preds, 0, num_classes - 1)
    labels = torch.clamp(labels, 0, num_classes - 1)

    iou = iou_metric(preds, labels)
    return iou.item()

def visualize_prediction(img_tensor, pred_mask, num_classes=19):
    img = img_tensor.cpu().permute(1, 2, 0).numpy()
    img = (img * list(STD)) + list(MEAN) # De-normalising
    img = (img * 255).astype(np.uint8)

    pred_mask = pred_mask.cpu().numpy()
    overlay = np.zeros_like(img)
    color_map = plt.get_cmap('nipy_spectral', num_classes)
    
    for cls in np.unique(pred_mask):
        rgb_image = color_map(cls)[:3]
        overlay[pred_mask == cls] = np.array(rgb_image) * 255

    blend = (0.5 * img + 0.5 * overlay).astype(np.uint8) # 50% of original image and 50% of the maskes.

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(blend)
    plt.title("Predicted Segmentation")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_optimizer(model, lr=6e-5):
    return optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

# A common function for freezing and unfreezing layers in 
# DeepLab, U-Net and SegFormer
def freeze_layers(model, should_freeze):
    should_require_grad = not should_freeze

    if should_freeze:
        print("Freezing layers...")
    else:
        print("Unfreezing layers...")

    if hasattr(model, 'backbone'):  # DeepLab
        for param in model.backbone.parameters():
            param.requires_grad = should_require_grad
    elif hasattr(model, 'encoder'):  # U-Net
        for param in model.encoder.parameters():
            param.requires_grad = should_require_grad
    elif hasattr(model, 'segformer'):  # SegFormer
        for param in model.segformer.encoder.parameters():
            param.requires_grad = should_require_grad

# # DeepLabV3
def train_deeplab(num_epochs, model, optimizer, train_loader, scheduler, freeze_epochs=5):

    model.train()
    total_loss = []
    start_time = time.time()
    freeze_layers(model, True)

    for epoch in range(num_epochs):

        if epoch == freeze_epochs:
            freeze_layers(model, False)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_start = time.time()
        train_loss = 0.0

        progress_bar = tqdm(train_loader,
                            desc= f"Epoch: {epoch + 1} / {num_epochs}",
                            unit="batch")

        for i, (images, labels) in enumerate(progress_bar, 1):
            images, labels = images.to(DEVICE), labels.to(DEVICE).long()
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = combined_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            progress_bar.set_postfix(avg_loss= train_loss/i)
        
        scheduler.step()

        avg_loss = train_loss / len(train_loader)
        total_loss.append(avg_loss)
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time/60:.2f} min | Avg Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    return total_loss

def evaluate_deeplab(model, val_loader, num_classes=19, visualize=True):
    print("\nEvaluating on validation set...")
    model.eval()

    # For pixel accuracy
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_iou = 0
    count = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validating")):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)

            # Flatten for confusion matrix (accuracy)
            preds_flat = preds.view(-1).cpu().numpy()
            labels_flat = labels.view(-1).cpu().numpy()
            mask = labels_flat != 255
            preds_flat = preds_flat[mask]
            labels_flat = labels_flat[mask]

            # Accuracy calculation using confusion matrix
            for p, t in zip(preds_flat, labels_flat):
                conf_matrix[t, p] += 1

            # mIoU using torchmetrics (requires torch tensors)
            iou = compute_iou(preds, labels)
            total_iou += iou
            count += 1

            # Visualization
            if visualize and batch_idx == 0:
                visualize_prediction(images[0], preds[0])

    # Compute pixel accuracy
    pixel_acc = np.diag(conf_matrix).sum() / conf_matrix.sum()

    # Compute mIoU using Jaccard Index

    miou = total_iou / count
    print(f"Validation mIoU: {miou:.4f}")
    print(f"Validation Pixel Accuracy: {pixel_acc:.4f}")

    return pixel_acc, miou

# # U-Net 
def train_unet(num_epochs, model, optimizer, train_loader, scheduler, freeze_epochs=5):

    model.train()
    total_loss = []
    start_time = time.time()
    freeze_layers(model, True)
    
    for epoch in range(num_epochs):
        if epoch == freeze_epochs:
            freeze_layers(model, False)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_start = time.time()
        train_loss = 0.0

        
        progress_bar = tqdm(train_loader,
                            desc= f"Epoch: {epoch + 1} / {num_epochs}",
                            unit="batch")

        for i, (images, labels) in enumerate(progress_bar, 1):
            images, labels = images.to(DEVICE), labels.to(DEVICE).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            progress_bar.set_postfix(avg_loss= train_loss/i)
        
        scheduler.step()

        avg_loss = train_loss / len(train_loader)
        total_loss.append(avg_loss)
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time/60:.2f} min | Avg Loss: {avg_loss:.4f}")

        
    total_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    return total_loss

def evaluate_unet(model, val_loader, num_classes=19, visualize=True):
    print("\nEvaluating on validation set...")
    model.eval()

    # For pixel accuracy
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_iou = 0
    count = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validating")):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Flatten for confusion matrix (accuracy)
            preds_flat = preds.view(-1).cpu().numpy()
            labels_flat = labels.view(-1).cpu().numpy()
            mask = labels_flat != 255
            preds_flat = preds_flat[mask]
            labels_flat = labels_flat[mask]

            # Accuracy calculation using confusion matrix
            for p, t in zip(preds_flat, labels_flat):
                conf_matrix[t, p] += 1

            iou = compute_iou(preds, labels)
            total_iou += iou
            count += 1

            if visualize and batch_idx == 0:
                visualize_prediction(images[0], preds[0])

    # Compute pixel accuracy
    pixel_acc = np.diag(conf_matrix).sum() / conf_matrix.sum()

    # Compute mIoU using Jaccard Index

    miou = total_iou / count
    print(f"Validation mIoU: {miou:.4f}")
    print(f"Validation Pixel Accuracy: {pixel_acc:.4f}")

    return pixel_acc, miou

# SegFormer
def train_segformer(num_epochs, model, optimizer, train_loader, scheduler, freeze_epochs=5):

    model.train()
    loss_history = []
    start_time = time.time()
    freeze_layers(model, True)

    for epoch in range(num_epochs):
        if epoch == freeze_epochs:
            freeze_layers(model, False)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_start = time.time()
        running_loss = 0.0
        progress_bar = tqdm(train_loader,
                            desc=f"Epoch: {epoch + 1} / {num_epochs}",
                            unit="batch")

        # Show current learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current Learning Rate: {current_lr:.8f}")

        for i, (images, labels) in enumerate(progress_bar, 1):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).long()

            optimizer.zero_grad()
            outputs = model(images)
            logits = outputs.logits

            logits = F.interpolate(logits, size=labels.shape[1:], 
                                   mode="bilinear", align_corners=False)

            loss = combined_loss(logits, labels)

            # Skip non-finite loss
            if not torch.isfinite(loss):
                print(f"Skipping batch due to non-finite loss: {loss.item()}")
                continue

            loss.backward()

            # Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(avg_loss=running_loss / i)
        
        # Step scheduler
        scheduler.step()

        avg = running_loss / len(train_loader)
        loss_history.append(avg)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time/60:.2f} min | Avg Loss: {avg:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    return loss_history

def evaluate_segformer(model, val_loader, num_classes=19, visualize=True):
    print("\nEvaluating on validation set v2...")
    model.eval()

    # For pixel accuracy
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_iou = 0
    count = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validating")):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images).logits

            # Resize predictions to match labels
            outputs = F.interpolate(outputs, size=labels.shape[1:], mode="bilinear", align_corners=False)
            preds = torch.argmax(outputs, dim=1)

            # Flatten for confusion matrix (accuracy)
            preds_flat = preds.view(-1).cpu().numpy()
            labels_flat = labels.view(-1).cpu().numpy()
            mask = labels_flat != 255
            preds_flat = preds_flat[mask]
            labels_flat = labels_flat[mask]

            # Accuracy calculation using confusion matrix
            for p, t in zip(preds_flat, labels_flat):
                conf_matrix[t, p] += 1

            iou = compute_iou(preds, labels)
            total_iou += iou
            count += 1
            

            # Visualization
            if visualize and batch_idx == 0:
                visualize_prediction(images[0], preds[0])

    # Compute pixel accuracy
    pixel_acc = np.diag(conf_matrix).sum() / conf_matrix.sum()

    miou = total_iou / count
    print(f"Validation mIoU: {miou:.4f}")
    print(f"Validation Pixel Accuracy: {pixel_acc:.4f}")

    return pixel_acc, miou

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Referenced from https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
def dice_loss(pred, target, smooth=1.0):
    pred = torch.softmax(pred, dim=1) 
    valid_mask = (target != 255)
    target_safe = target.clone()
    target_safe[~valid_mask] = 0
    target_onehot = F.one_hot(target_safe, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(outputs, labels):
    focal_loss = FocalLoss(ignore_index=255)
    f_loss = focal_loss(outputs, labels)
    d_loss = dice_loss(outputs, labels)
    return 0.5 * f_loss + 0.5 * d_loss

class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, end_lr=0.0, power=0.9, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.end_lr = end_lr
        self.power = power
        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch + 1
        if current_epoch <= self.warmup_epochs:
            # Linear warmup
            return [1e-6 + (self.base_lr - 1e-6) * current_epoch / self.warmup_epochs for _ in self.base_lrs]
        else:
            # Poly decay. Obtained from https://arxiv.org/pdf/1706.05587 in the 4.1 Training Protocol section
            iter = current_epoch - self.warmup_epochs
            max_iter = self.total_epochs - self.warmup_epochs
            progress = iter / max_iter
            poly_factor = (1 - progress) ** self.power
            return [self.end_lr + (self.base_lr - self.end_lr) * poly_factor for _ in self.base_lrs]

def getDeviceType():
    # Checking the DEVICE to run
    if torch.cuda.is_available():
        return torch.device("cuda:1")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

if __name__ == "__main__":

    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    DEVICE = getDeviceType()
    print("DEVICE:", DEVICE)
    EPOCHS = 50
    RESIZE = (512,512)
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    FLIP_RATIO = 0.5
    BRIGHTNESS = 0.2
    NUM_CLASSES = 19

    image_transform = T.Compose([
        T.Resize(RESIZE),
        T.RandomHorizontalFlip(p= FLIP_RATIO),
        T.ColorJitter(brightness= BRIGHTNESS, contrast= BRIGHTNESS),
        T.ToTensor(),
        T.Normalize(mean= MEAN, std= STD),
    ])

    label_transform = T.Compose([
        T.Resize(RESIZE, interpolation=Image.Resampling.NEAREST),
        T.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
    ])

    train_dataset = CityscapesDataset(
        'leftImg8bit',
        'gtFine',
        split='train',
        transform=image_transform,
        target_transform=label_transform
    )

    val_dataset = CityscapesDataset(
        'leftImg8bit',
        'gtFine',
        split='val',
        transform=image_transform,
        target_transform=label_transform
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    images, labels = next(iter(train_loader))
    print("Images Shape: ", images.shape)
    print("Labels Shape: ", labels.shape)
    print("Unique Labels: ", np.unique(labels))

    # DeepLabV3
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    deeplab_model = deeplabv3_resnet101(weights=weights)
    deeplab_model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    deeplab_model = deeplab_model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = create_optimizer(deeplab_model)

    base_lr = 6e-5
    warmup_epochs = 5
    scheduler = WarmupPolyLR(optimizer, warmup_epochs, EPOCHS, base_lr)

    print("------------------------------------------")
    print("Training DeepLabV3 started...")
    deeplab_total_loss = train_deeplab(EPOCHS, deeplab_model, optimizer, train_loader,scheduler)
    print("Training DeepLabV3 ended...")
    print("DeepLabV3 Total Loss: ", deeplab_total_loss)
    print("Validating DeepLabV3 started...")
    deeplab_accuracy, deeplab_miou = evaluate_deeplab(deeplab_model, val_loader)
    print("Validating DeepLabV3 ended...")


    torch.cuda.empty_cache()
    torch.save(deeplab_model.state_dict(), "deeplabv3_resnet101_run11.pth")
    print("Model saved as deeplabv3_resnet101_run11.pth")


    # U-Net
    unet_model = smp.Unet(
        encoder_name="resnet101",       
        encoder_weights="imagenet",   
        in_channels=3,
        classes= NUM_CLASSES                   
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = create_optimizer(unet_model)

    base_lr = 6e-5
    warmup_epochs = 5
    scheduler = WarmupPolyLR(optimizer, warmup_epochs, EPOCHS, base_lr)

    print("------------------------------------------")
    print("Training U-Net started...")
    unet_total_loss = train_unet(EPOCHS, unet_model, optimizer, train_loader,scheduler)
    print("Training U-Net ended...")
    print("U-Net Total Loss: ", unet_total_loss)
    print("Validating U-Net started...")
    unet_accuracy, unet_miou = evaluate_unet(unet_model, val_loader)
    print("Validating U-Net ended...")

    torch.cuda.empty_cache()
    torch.save(unet_model.state_dict(), "unet_model_resnet101_run11.pth")
    print("Model saved as unet_model_resnet101_run11.pth")

    # SegFormer
    segformer_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",   
        num_labels= NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = create_optimizer(segformer_model)

    base_lr = 6e-5
    warmup_epochs = 5
    scheduler = WarmupPolyLR(optimizer, warmup_epochs, EPOCHS, base_lr)

    print("------------------------------------------")
    print("Training SegFormer started...")
    segformer_total_loss = train_segformer(EPOCHS, segformer_model, optimizer, train_loader, scheduler)
    print("Training SegFormer ended...")
    print("SegFormer Total Loss: ", segformer_total_loss)
    print("Validating SegFormer started...")
    segformer_accuracy, segformer_miou = evaluate_segformer(segformer_model, val_loader)
    print("Validating SegFormer ended...")

    torch.cuda.empty_cache()
    torch.save(segformer_model.state_dict(), "segformer_model_b2_run11.pth")
    print("Model saved as segformer_model_b2_run11.pth")