import torch
import time
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import albumentations as A
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import LambdaLR
from transformers import SegformerForSemanticSegmentation
from torchmetrics.classification import MulticlassJaccardIndex


class FoodSegDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        original_image = self.dataset[index]
        image = np.array(original_image['image'])
        mask = np.array(original_image['label'], dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image= image, mask= mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()  # Returns as ``long - dtype`` as CrossEntropy expects a long type value
    
def compute_iou(preds, labels, num_classes=104, ignore_index=255):
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

def visualize_prediction(img_tensor, pred_mask, num_classes=104):
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

def poly_lr_lambda(epoch, EPOCHS = 100, power=0.9):
    return (1 - epoch / EPOCHS) ** power

def create_optimizer(model, lr=6e-5):
    return optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

# SegFormer
def train_segformer(num_epochs, model, optimizer, train_loader, scheduler):
    loss_history = []
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        epoch_start = time.time()
        running_loss = 0.0
        progress_bar = tqdm(train_loader,
                            desc=f"Epoch: {epoch + 1} / {num_epochs}",
                            unit="batch")

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
        
        scheduler.step()

        avg = running_loss / len(train_loader)
        loss_history.append(avg)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time/60:.2f} min | Avg Loss: {avg:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    return loss_history

def evaluate_segformer(model, val_loader, num_classes=104, visualize=True):
    print("\nEvaluating on validation set v2...")
    model.eval()

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

            preds_flat = preds.view(-1).cpu().numpy()
            labels_flat = labels.view(-1).cpu().numpy()
            mask = labels_flat != 255
            preds_flat = preds_flat[mask]
            labels_flat = labels_flat[mask]

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

# Referenced from https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
def dice_loss(pred, target, smooth=1.0):
    pred = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(outputs, labels):
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    ce = criterion(outputs, labels)
    dice = dice_loss(outputs, labels)
    return 0.7 * ce + 0.3 * dice

def getDeviceType():
    # Checking the DEVICE to run
    if torch.cuda.is_available():
        return torch.device("cuda:1")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

if __name__ == "__main__":
    original_dataset = load_dataset("EduardoPacheco/FoodSeg103")
    original_dataset

    train_dataset = original_dataset["train"]
    val_dataset = original_dataset["validation"]

    print("Images size: ", train_dataset['image'][0].size)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))


    DEVICE = getDeviceType()
    print("DEVICE:", DEVICE)
    EPOCHS = 100
    HEIGHT = 512
    WIDTH = 384
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    FLIP_RATIO = 0.5
    BRIGHTNESS = 0.2
    NUM_CLASSES = 104

    train_transforms = A.Compose([
        A.Resize(HEIGHT, WIDTH),
        A.HorizontalFlip(p= FLIP_RATIO),
        A.RandomBrightnessContrast(p= BRIGHTNESS),
        A.Normalize(mean= MEAN, std= STD),
        ToTensorV2()
    ])

    val_transforms = A.Compose([
        A.Resize(HEIGHT, WIDTH),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

    training_dataset = FoodSegDataset(dataset= train_dataset, transform= train_transforms)
    validation_dataset = FoodSegDataset(dataset= val_dataset, transform= val_transforms)

    train_dataloader = torch.utils.data.DataLoader(training_dataset,
                                                   batch_size=8,
                                                   shuffle= True)

    val_dataloader = torch.utils.data.DataLoader(validation_dataset,
                                                batch_size=4,
                                                shuffle= False)
    
    images, labels = next(iter(train_dataloader))
    print("Images Shape: ", images.shape)
    print("Labels Shape: ", labels.shape)
    print("Unique Labels: ", np.unique(labels))

    # SegFormer
    segformer_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",   
        num_labels= NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    optimizer = create_optimizer(segformer_model)
    scheduler = LambdaLR(optimizer, lr_lambda=poly_lr_lambda)

    print("------------------------------------------")
    print("Training SegFormer started...")
    segformer_total_loss = train_segformer(EPOCHS, segformer_model, optimizer, train_dataloader, scheduler)
    print("Training SegFormer ended...")
    print("SegFormer Total Loss: ", segformer_total_loss)
    print("Validating SegFormer started...")
    segformer_accuracy, segformer_miou = evaluate_segformer(segformer_model, val_dataloader)
    print("Validating SegFormer ended...")

    torch.cuda.empty_cache()
    torch.save(segformer_model.state_dict(), "segformer_model_b2_run4.pth")
    print("Model saved as segformer_model_b2_run4.pth")