Facial Emotion Recognition (FER) with Deep Learning

End-to-End Facial Emotion Recognition pipeline using CNN and MLP on the ICML Face Dataset (48x48 grayscale images, 7 emotions).

✨ Features
Model           Architecture                  Key Techniques

CNN             Conv2D(48) + Dense(96)        Data Augmentation, Class Weights, Dropout

MLP             Dense(128→64→7)               PCA Dimensionality Reduction

PCA             98% Variance                  2304→~100 components

Emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

📊 Dataset

ICML 2013 Face Dataset

Total: ~35,887 images
Train: 28,709 (80%)
Val: 3,589 (10%)
Test: 3,589 (10%)
Image Size: 48×48 grayscale
Classes: 7 imbalanced emotions

🚀 Quick Start
Prerequisites

pip install tensorflow pandas numpy matplotlib seaborn scikit-learn mlxtend imbalanced-learn
Usage

jupyter notebook fer_analysis.ipynb
# Download: icml_face_data.csv (~100MB)

🛠️ Pipeline Overview

1. Data Loading → 2. Cleaning → 3. Preprocessing → 4. EDA → 5. Modeling → 6. Evaluation
1. Data Quality

✅ No missing values
✅ Duplicates removed
✅ Pixels parsed: string → 48×48 arrays
✅ Labels: 0-6 → one-hot encoded
2. Preprocessing Pipeline

# Image reshaping: (N, 48, 48) → (N, 48, 48, 1)
# Normalization: uint8 → float32 [0,1]
# One-hot encoding: 7 classes
# Class weights: Handle imbalance
# PCA: 98% variance retention
3. Exploratory Analysis

📊 Emotion Distribution: Happy(25%) > Neutral(19%) > Sad(15%)
🖼️ Sample Visualization: 7 emotion examples
🎨 Train vs Val distributions compared
4. Model Architectures

CNN Model (Primary)

Input: 48×48×1
Conv2D(48, 3×3) ×2 → MaxPool → Dropout(0.2)
Flatten → Dense(96) ×2 → Dropout(0.2)
Output: Softmax(7)
Optimizer: Adam(lr=1e-3)

MLP Model (Baseline)

Input: 2304 pixels (flattened)
Dense(128) → Dense(64) → Softmax(7)
PCA + RandomForest

Components: ~100 (98% variance)
RF: n_estimators=500
📈 Performance Results

Model               Test Accuracy               Train/Val Curves              Notes

CNN                 ~65-70%                       Converging              Best performance

MLP                 ~55-60%                       Overfitting             Baseline

RF+PCA              Calculated                     N/A                    Dimensionality reduction

Training History: Loss/Accuracy curves plotted

🎨 Key Visualizations

1. Emotion Distribution (Train vs Val)
2. Sample Images (7 emotions)
3. Training Curves (Loss + Accuracy)
4. Confusion Matrix (mlxtend)
5. PCA Scatter Plot (PC1 vs PC2)

🔬 Advanced Techniques

Class Imbalance Handling

class_weight = {0:0.14, 1:0.03, 2:0.08, 3:0.25, 4:0.15, 5:0.07, 6:0.19}

PCA Dimensionality Reduction

Original: 2,304 pixels → ~100 components (98% variance)
Cumulative Variance: [0.25, 0.45, 0.62, ... , 0.98]

Data Augmentation Ready

ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True)
🧪 Sample Code & Outputs
# Model Summary
CNN: 289,815 params (trainable)
MLP: ~150K params

# Evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy:', test_acc)

# Confusion Matrix
plot_confusion_matrix(conf_mat, figsize=(10,8))

📁 Expected Results

Epoch 10/10:
Train Acc: ~68%
Val Acc: ~62%
Test Acc: ~65%

Top Predictions:
Happy: 72% accuracy
Neutral: 68%
Sad: 61%

🔧 Customization
# CNN Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

# PCA
VARIANCE_THRESHOLD = 0.98
N_COMPONENTS = 100

# Class Weights (adjust for imbalance)

👩‍💻 Author
Nourhan Mohammed
Computer Science Student | Data Enthusiast
