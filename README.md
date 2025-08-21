# Writer Recognition System

A deep learning-based system for identifying and classifying handwritten text by different writers using computer vision and neural networks.

## 📋 Project Overview

This project implements a **Writer Recognition System** that can automatically identify the author of handwritten text samples. It uses deep learning techniques to analyze handwriting characteristics and classify text samples by their writers with high accuracy.

## 🎯 Features

- **Writer Identification**: Automatically identify the author of handwritten text
- **Deep Learning Model**: Built with PyTorch using ResNet18 architecture
- **Data Filtering**: Intelligent dataset filtering to ensure quality training data
- **Data Augmentation**: Image transformations for improved model robustness
- **Per-Writer Split**: Maintains writer integrity in train/test splits
- **GPU Acceleration**: CUDA support for faster training and inference

## 🏗️ Architecture

The system uses a **ResNet18** backbone modified for grayscale handwriting images:

- **Input**: Grayscale handwriting images (224x224 pixels)
- **Backbone**: ResNet18 with ImageNet pretrained weights
- **Modifications**: 
  - First convolutional layer adapted for grayscale (1 channel)
  - Final classification layer for writer identification
- **Output**: Writer class probabilities

## 📁 Project Structure

```
DLP-Project-Writer-Recognizer/
├── Writer_Recognizer.ipynb          # Main Jupyter notebook with implementation
├── DLP_PROJECT_[22K-4316, 22K-4369, 22K-4303].pdf  # Project documentation
├── .gitattributes                   # Git configuration
└── README.md                        # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- PIL (Pillow)
- tqdm
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd DLP-Project-Writer-Recognizer
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision pillow tqdm
   ```

3. **Prepare your dataset**:
   - Organize handwriting images in folders by writer
   - Each writer should have their own folder with multiple samples
   - Supported formats: JPG, PNG, JPEG

### Dataset Structure

```
data/
├── writer_1/
│   ├── sample1.jpg
│   ├── sample2.png
│   └── ...
├── writer_2/
│   ├── sample1.jpg
│   └── ...
└── ...
```

## 💻 Usage

### Basic Usage

1. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook Writer_Recognizer.ipynb
   ```

2. **Update the data path** in the notebook:
   ```python
   DATA_DIR = "/path/to/your/handwriting/dataset"
   ```

3. **Run the training pipeline**:
   ```python
   run_pipeline(DATA_DIR)
   ```

### Training Process

The system automatically:
- Filters writers with insufficient samples (minimum 5 images per writer)
- Splits data into training (80%) and testing (20%) sets
- Applies data augmentation (resize, flip, rotation)
- Trains the model for 50 epochs
- Evaluates performance on test set
- Saves the trained model

### Model Training

```python
# Training parameters
epochs = 50
learning_rate = 1e-4
batch_size = 16

# Data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

## 🔧 Key Components

### 1. FilteredWriterDataset
- Filters writers with minimum image requirements
- Ensures balanced training data
- Handles grayscale image conversion

### 2. WriterClassifier
- Custom neural network architecture
- ResNet18 backbone with modifications
- Cosine similarity loss for better writer discrimination

### 3. Training Pipeline
- Automatic data splitting
- Balanced sampling
- Progress tracking with tqdm
- Model checkpointing

## 📊 Performance

The system typically achieves:
- **Training Loss**: Decreases from ~150 to ~35 over 50 epochs
- **Test Accuracy**: Varies based on dataset quality and writer count
- **Training Time**: ~1.25 seconds per epoch on GPU

## 🎨 Data Augmentation

To improve model robustness, the system applies:
- **Resize**: Standardize image dimensions to 224x224
- **Random Horizontal Flip**: Simulate different writing orientations
- **Random Rotation**: Handle slight writing angle variations
- **Normalization**: Standardize pixel values for better training

## 💾 Model Saving

Trained models are automatically saved as:
- `writer_id_model.pth` (unfiltered dataset)
- `writer_classifier_cosine.pth` (filtered dataset)

## 🔍 Inference

After training, you can use the model for writer identification:

```python
# Load trained model
model = WriterClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load("writer_classifier_cosine.pth"))

# Predict writer for new image
prediction = model(image_tensor)
predicted_writer = class_names[prediction.argmax().item()]
```

## 🛠️ Customization

### Modify Training Parameters
```python
# Adjust training settings
epochs = 100          # More training epochs
learning_rate = 5e-5  # Lower learning rate
batch_size = 32       # Larger batch size
```

### Change Data Augmentation
```python
# Custom transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),      # Different image size
    transforms.ColorJitter(brightness=0.2),  # Add color variation
    transforms.RandomAffine(degrees=15),     # More rotation
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image dimensions
2. **Low Accuracy**: Check dataset quality and increase training epochs
3. **Slow Training**: Ensure GPU is available and CUDA is properly installed

### Performance Tips

- Use GPU acceleration when available
- Ensure sufficient RAM for data loading
- Consider data preprocessing for large datasets

## 📚 Technical Details

### Model Architecture
- **Base Model**: ResNet18 (ImageNet pretrained)
- **Input Layer**: Modified for grayscale (1 channel)
- **Output Layer**: Custom classification head
- **Loss Function**: CrossEntropyLoss with label smoothing

### Data Processing
- **Image Format**: Grayscale (L mode)
- **Dimensions**: 224x224 pixels
- **Normalization**: Mean=0.5, Std=0.5
- **Augmentation**: Random flip, rotation, resize

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is part of a DLP (Deep Learning Project) course assignment.

## 👥 Authors

- **22K-4316**
- **22K-4369** 
- **22K-4303**

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- ResNet architecture developers
- Course instructors and teaching assistants

---

**Note**: This project is designed for educational purposes and research in writer recognition. For production use, additional validation and testing is recommended.
