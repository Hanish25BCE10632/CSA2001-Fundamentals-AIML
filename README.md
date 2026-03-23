# CSA2001 - Fundamentals in AI & ML

## BYOP Project – Offroad Semantic Segmentation using DINOv2

---

## Student Details

* **Name:** Hanish Singla
* **Registration Number:** 25BCE10632
* **Course Code:** CSA2001
* **Course Title:** Fundamentals in AI and ML
* **Branch:** Computer Science and Engineering
* **Year:** First Year
* **Date:** 22 March 2026

---

## Project Overview

This project focuses on **Semantic Segmentation**, a computer vision task where each pixel of an image is classified into a specific category.

The goal is to segment **off-road scenes** into multiple classes such as:

* Trees
* Rocks
* Sky
* Ground

We implemented a **deep learning pipeline** using a **pre-trained DINOv2 backbone** and a custom segmentation head.

---

## 🎯 Objectives

* Perform pixel-wise classification of images
* Train and evaluate a segmentation model
* Compare performance across different training configurations
* Visualize predictions using color-coded outputs

---

## ⚙️ Tech Stack

* Python 3.13.9
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* DINOv2 (Vision Transformer)

---

## 📂 Project Structure

```
CSA2001-Fundamentals-AIML/
│
├── data/
│   ├── train/
│   │   ├── Color_Images/
│   │   └── Segmentation/
│   │
│   ├── val/
│       ├── Color_Images/
│       └── Segmentation/
│
├── Offroad_Segmentation_testImages/
│   ├── Color_Images/
│   └── Segmentation/
│
├── scripts/
│   ├── train_segmentation.py
│   ├── test_segmentation.py
│   ├── visualize.py        
│   ├── segmentation_head.pth
│   └── train_stats/
│
├── predictions/
│   ├── masks/
│   ├── masks_color/
│   ├── comparisons/
│   ├── evaluation_metrics.txt
│   └── per_class_metrics.png
│
├── requirements.txt
├── README.md
└── report.pdf
```

---

## 🛠️ Environment Setup

```bash
conda create -n EDU python=3.13.9
conda activate EDU

pip install torch torchvision numpy matplotlib opencv-python tqdm Pillow
pip install -r requirements.txt    
```

---

## 🧪 Training

```bash
cd scripts
python train_segmentation.py
```

---

## 🧪 Testing

```bash
python test_segmentation.py --data_dir ../Offroad_Segmentation_testImages --output_dir ../predictions
```

---

## 🧪 Visualize 

```bash
python visualize.py
```

---

## Experiments Conducted

### 🔹 Experiment 1: 10 Epochs

* Validation IoU: ~0.324
* Test IoU: ~0.226
* Accuracy: ~70%

### 🔹 Experiment 2: 25 Epochs (Final Model)

* Validation IoU: ~0.326
* Test IoU: ~0.233
* Accuracy: ~70%

---

## Detailed Training Insights

The model was trained for 25 epochs and showed consistent improvement in performance.

- Initial Val IoU (Epoch 1): ~0.21
- Final Val IoU (Epoch 25): ~0.326

- Training Loss reduced from ~1.36 to ~0.80
- Validation Loss reduced from ~1.10 to ~0.80

This shows that the model effectively learned meaningful features over time.

The best performance was achieved at Epoch 25, indicating stable convergence without overfitting.

---

## Observation

Increasing the number of epochs from 10 to 25 resulted in a gradual improvement in performance, particularly in IoU and loss reduction.

However, the improvement was relatively small, indicating that the model had already started to converge after early epochs.

The training and validation metrics followed similar trends, suggesting that overfitting was minimal and generalization was stable.

---

## Evaluation Metrics

* **IoU (Intersection over Union)**
* **Dice Score**
* **Pixel Accuracy**

---

## Outputs

The model generates:

* Colorized segmentation masks
* Side-by-side comparisons
* Evaluation metrics
* Training graphs

📁 Output directory:

```
predictions/
 ├── masks/
 ├── masks_color/
 ├── comparisons/
 ├── evaluation_metrics.txt
```

---

## Key Learnings

* Understanding of semantic segmentation
* Use of pre-trained models (transfer learning)
* Model evaluation using IoU
* Effect of hyperparameters like epochs
* Real-world AI pipeline implementation

---

## Challenges Faced

* Long training time on CPU
* Dataset handling and preprocessing
* Model tuning and optimization

---

## Future Improvements

* Use GPU acceleration
* Apply data augmentation
* Use advanced architectures (U-Net, DeepLabV3+)
* Hyperparameter tuning

---

## Conclusion

This project demonstrates how **AI and Computer Vision** can be used for real-world scene understanding.
The model successfully segments off-road environments and provides meaningful visual outputs.

---
