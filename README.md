# 🍽️ Semantic Segmentation for Food Images using DeepLabV3, U-Net, and SegFormer

This repository presents a comprehensive comparative study of three popular semantic segmentation architectures — **DeepLabV3**, **U-Net**, and **SegFormer** — trained and evaluated on the **FoodSeg103** dataset. The goal of this project is to analyze and compare their effectiveness in segmenting complex food items at a pixel level using various training strategies, loss functions, and learning rate schedules.

---

## 🧪 Experiments Conducted

A total of **10 experiments** were conducted, each with different configurations:

- **Experiment 1:** Baseline with CrossEntropyLoss
- **Experiment 2:** Paper-based hyperparameters (Sinha et al., 2023)
- **Experiment 3:** CrossEntropy + Dice Loss combination
- **Experiment 4–6:** Variations with 100 epochs and `WarmupPolyLR` scheduler
- **Experiment 7–8:** Freezing/Unfreezing encoder layers + Focal + Dice Loss
- **Experiment 9–10:** U-Net++ and generalization to Cityscapes dataset

---

## 📁 Dataset: [FoodSeg103](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103)

- Real-world food images
- 104 semantic classes (103 food ingredients + background)
- Pixel-level annotations
- Challenges include overlapping items, class imbalance, and fine-grained boundaries

---

## 🧠 Model Architectures Used

| Model       | Backbone       | Pretraining        |
|-------------|----------------|--------------------|
| DeepLabV3   | ResNet-101     | COCO / ImageNet    |
| U-Net       | ResNet-101     | ImageNet           |
| SegFormer   | SegFormer-B2   | ADE20K             |
| U-Net++     | ResNet-101     | ImageNet           |

All models were adapted using PyTorch and [Segmentation Models PyTorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch) or HuggingFace Transformers.

---

### 📊 Experimental Results (FoodSeg103)

| Exp | DeepLab Accuracy (%) | U-Net Accuracy (%) | SegFormer Accuracy (%) | DeepLab mIoU (%) | U-Net mIoU (%) | SegFormer mIoU (%) |
|-----|----------------------|--------------------|------------------------|------------------|----------------|---------------------|
| 1   | **77.84**            | 75.29              | 77.74                  | 26.83            | 16.32          | **28.82**           |
| 2   | 77.99                | 75.00              | **78.80**              | 27.37            | 16.73          | **32.00**           |
| 3   | **78.14**            | 74.67              | 77.91                  | 32.37            | 21.46          | **37.18**           |
| 4   | -                    | -                  | **79.08**              | -                | -              | **36.57**           |
| 5   | -                    | -                  | **78.55**              | -                | -              | **38.22**           |
| 6   | -                    | -                  | **78.86**              | -                | -              | **36.80**           |
| 7   | 78.47                | 75.00              | **79.87**              | 33.36            | 19.90          | **40.76**           |
| 8   | 77.73                | 75.21              | **79.18**              | 35.51            | 22.71          | **41.14**           |
| 9   | 77.47                | 74.89              | **78.70**              | 34.48            | 23.73          | **40.39**           |
| 10  | -                    | **75.59**          | -                      | -                | 23.68          | -                   |
