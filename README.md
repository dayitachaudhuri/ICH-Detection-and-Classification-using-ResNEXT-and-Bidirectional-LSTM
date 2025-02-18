# Intracranial Hemorrhage Detection and Classification

## Overview
This project implements an ensemble deep learning framework for automated detection and classification of Intracranial Hemorrhage (ICH) from brain CT scans. The system combines SE-ResNeXT and bidirectional LSTM architectures to process and analyze brain CT images, capable of detecting and classifying five different ICH subtypes:
- Intraparenchymal (IPH)
- Intraventricular (IVH)
- Subarachnoid (SAH)
- Subdural (SDH)
- Epidural (EDH)

## Features
- Multi-label classification for hemorrhage detection and subtype classification
- Advanced preprocessing of DICOM images
- Integration of multiple CT window settings
- Data augmentation for improved model generalization
- Grad-CAM visualization for model interpretability
- Ensemble approach combining CNN and LSTM architectures

## Requirements
- Python 3.x
- PyTorch
- torchvision
- NumPy
- pandas
- pydicom
- albumentations
- OpenCV
- scikit-learn

## Installation
```bash
# Clone the repository
git clone https://github.com/dayitachaudhuri/ICH-Detection-and-Classification-using-ResNEXT-and-Bidirectional-LSTM
cd ICH-Detection-and-Classification-using-ResNEXT-and-Bidirectional-LSTM

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preprocessing
The system processes DICOM format CT scans using three different window settings:
- Brain window (WC = 40, WW = 80)
- Subdural window (WC = 80, WW = 200)
- Soft tissue window (WC = 40, WW = 380)

Images are preprocessed by:
1. Converting DICOM to Hounsfield units
2. Applying window settings
3. Resizing to 256×256 pixels
4. Normalizing using preset mean and standard deviation values
5. Applying data augmentation techniques

## Model Architecture
### CNN Component (SE-ResNeXT)
- Base model: ResNeXt-101 32×8d
- Pre-trained on 940 million public images
- Modified classification layer for hemorrhage detection

### LSTM Component
- 3-layer bidirectional LSTM
- 256-dimensional embeddings
- Dropout rate: 0.3
- Processes sequences of CNN-generated features

## Training
Key training parameters:
- Optimizer: Adam
- CNN learning rates:
  - Epochs 1-2: 1e-4
  - Epoch 3: 2e-5
- LSTM learning rate: 1e-4
- Loss function: Multi-label binary cross-entropy

## Performance Metrics
The model achieves the following performance metrics on the RSNA dataset:

| Metric | Slice-level | Scan-level |
|--------|-------------|------------|
| Accuracy | 0.9412 | 0.9296 |
| Sensitivity | 0.7763 | 0.8313 |
| Specificity | 0.9690 | 0.9477 |

## Usage
### For Prediction
```python
from models.ensemble import ICHEnsemble
from utils.preprocessing import preprocess_scan

# Load model
model = ICHEnsemble.load_from_checkpoint('path/to/checkpoint.ckpt')

# Preprocess and predict
scan = preprocess_scan('path/to/dicom/files')
predictions = model.predict(scan)
```

### For Visualization
```python
from utils.visualization import generate_gradcam

# Generate Grad-CAM visualization
gradcam_map = generate_gradcam(model, image, target_class)
```

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- RSNA for providing the Brain CT Hemorrhage Dataset
- Implementation based on the work by Burduja et al.
