import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import albumentations as A
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import LambdaLR
from transformers import SegformerForSemanticSegmentation
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights


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

def poly_lr_lambda(epoch, max_epochs= 50, power=0.9):
    return (1 - epoch / max_epochs) ** power

def create_optimizer(model, lr=6e-5):
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

# DeepLabV3
def train_deeplab(num_epochs, model, optimizer, criterion, train_loader, scheduler):

    model.train()
    total_loss = []
    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = 0.0

        progress_bar = tqdm(train_loader,
                            desc= f"Epoch: {epoch + 1} / {num_epochs}",
                            unit="batch")

        for i, (images, labels) in enumerate(progress_bar, 1):
            images, labels = images.to(DEVICE), labels.to(DEVICE).long()
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            progress_bar.set_postfix(avg_loss= train_loss/i)
        
        scheduler.step()

        avg_loss = train_loss / len(train_loader)
        total_loss.append(avg_loss)
        print(f"Train Loss: {avg_loss:.4f}")

    return total_loss

def evaluate_deeplab(model, val_loader, num_classes=104, visualize=True):
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

# U-Net 
def train_unet(num_epochs, model, optimizer, criterion, train_loader, scheduler):

    model.train()
    total_loss = []
    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = 0.0

        
        progress_bar = tqdm(train_loader,
                            desc= f"Epoch: {epoch + 1} / {num_epochs}",
                            unit="batch")

        for i, (images, labels) in enumerate(progress_bar, 1):
            images, labels = images.to(DEVICE), labels.to(DEVICE).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            progress_bar.set_postfix(avg_loss= train_loss/i)
        
        scheduler.step()

        avg_loss = train_loss / len(train_loader)
        total_loss.append(avg_loss)
        print(f"Train Loss: {avg_loss:.4f}")

    return total_loss

def evaluate_unet(model, val_loader, num_classes=104, visualize=True):
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
    miou = total_iou / count
    print(f"Validation mIoU: {miou:.4f}")
    print(f"Validation Pixel Accuracy: {pixel_acc:.4f}")

    return pixel_acc, miou

# SegFormer
def train_segformer(num_epochs, model, optimizer, criterion, train_loader, scheduler):
    loss_history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        running_loss = 0.0
        progress_bar = tqdm(train_loader,
                            desc= f"Epoch: {epoch + 1} / {num_epochs}",
                            unit="batch")

        for i, (images, labels) in enumerate(progress_bar, 1):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).long()

            optimizer.zero_grad()
            outputs = model(images)
            logits = outputs.logits

            logits = F.interpolate(logits, size=labels.shape[1:], 
                                   mode="bilinear", align_corners=False)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(avg_loss=running_loss/i)
        
        scheduler.step()

        avg = running_loss / len(train_loader)
        loss_history.append(avg)
        print(f"Epoch {epoch+1} Avg Loss: {avg:.4f}")

    return loss_history

def evaluate_segformer(model, val_loader, num_classes=104, visualize=True):
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
    EPOCHS = 50
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
        A.Normalize(mean= MEAN, std= STD),
        ToTensorV2()
    ])

    training_dataset = FoodSegDataset(dataset= train_dataset, 
                                      transform= train_transforms)
    
    validation_dataset = FoodSegDataset(dataset= val_dataset, 
                                        transform= val_transforms)

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

    # DeepLabV3
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    deeplab_model = deeplabv3_resnet101(weights=weights)
    deeplab_model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    deeplab_model = deeplab_model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = create_optimizer(deeplab_model)
    scheduler = LambdaLR(optimizer, lr_lambda=poly_lr_lambda)

    print("------------------------------------------")
    print("Training DeepLabV3 started...")
    deeplab_total_loss = train_deeplab(EPOCHS, deeplab_model, optimizer, criterion, train_dataloader,scheduler)
    print("Training DeepLabV3 ended...")
    print("DeepLabV3 Total Loss: ", deeplab_total_loss)
    print("Validating DeepLabV3 started...")
    deeplab_accuracy, deeplab_miou = evaluate_deeplab(deeplab_model, val_dataloader)
    print("Validating DeepLabV3 ended...")

    torch.cuda.empty_cache()
    torch.save(deeplab_model.state_dict(), "deeplabv3_resnet101_run2.pth")
    print("Model saved as deeplabv3_resnet101_run2.pth")


    # U-Net
    unet_model = smp.Unet(
        encoder_name="resnet101",       
        encoder_weights="imagenet",     # pretrained on ImageNet
        in_channels=3,
        classes= NUM_CLASSES                   
    )

    unet_model = unet_model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = create_optimizer(unet_model)
    scheduler = LambdaLR(optimizer, lr_lambda=poly_lr_lambda)

    print("------------------------------------------")
    print("Training U-Net started...")
    unet_total_loss = train_unet(EPOCHS, unet_model, optimizer, criterion, train_dataloader,scheduler)
    print("Training U-Net ended...")
    print("U-Net Total Loss: ", unet_total_loss)
    print("Validating U-Net started...")
    unet_accuracy, unet_miou = evaluate_unet(unet_model, val_dataloader)
    print("Validating U-Net ended...")

    torch.cuda.empty_cache()
    torch.save(unet_model.state_dict(), "unet_model_resnet101_run2.pth")
    print("Model saved as unet_model_resnet101_run2.pth")

    # SegFormer
    segformer_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",   
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )

    segformer_model = segformer_model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = create_optimizer(segformer_model)
    scheduler = LambdaLR(optimizer, lr_lambda=poly_lr_lambda)

    print("------------------------------------------")
    print("Training SegFormer started...")
    segformer_total_loss = train_segformer(EPOCHS, segformer_model, optimizer,criterion, train_dataloader, scheduler)
    print("Training SegFormer ended...")
    print("SegFormer Total Loss: ", segformer_total_loss)
    print("Validating SegFormer started...")
    segformer_accuracy, segformer_miou = evaluate_segformer(segformer_model, val_dataloader)
    print("Validating SegFormer ended...")

    torch.cuda.empty_cache()
    torch.save(segformer_model.state_dict(), "segformer_model_b2_run2.pth")
    print("Model saved as segformer_model_b2_run2.pth")