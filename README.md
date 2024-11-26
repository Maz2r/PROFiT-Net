
# Deep Learning Project: CNN-based Prediction Framework

This project implements a framework for training and evaluating convolutional neural networks (CNNs) using Keras. It supports multiple target labels and includes data preprocessing, training, and prediction utilities.

## Features
- **Data Preprocessing**:
  - Extracts `.tar` files and processes data into `.csv` and `.npy` formats.
  - Splits data into training, validation, and test sets.

- **Model Training**:
  - Train CNN models for specific target labels.
  - Supports model checkpointing for resuming training or prediction.

- **Prediction**:
  - Load pre-trained models to make predictions on test data.
  - Provides performance statistics (MAE, MSE, R²) after predictions.

## Project Structure
```
deep_learning_project/
├── src/
│   ├── data/
│   │   ├── preprocess_data.py      # Script for preprocessing .tar files
│   │   ├── data_loader.py          # DataLoader for loading .npy files
│   ├── models/
│   │   ├── keras/
│   │   │   ├── cnn_model.py        # CNN model definition for Keras
│   ├── train/
│   │   ├── keras/
│   │   │   ├── train.py            # Training script for Keras
│   ├── predict/
│   │   ├── keras/
│   │   │   ├── predict.py          # Prediction script for Keras
│   ├── utils/
│   │   ├── callbacks.py            # Custom callbacks for training
│   │   ├── target_labels.py        # Abbreviation and mapping of target labels
├── tests/                          # Unit tests for components
├── data/                           # Directory for processed .npy files
├── dataset/                        # Directory for extracted .csv files
├── callback/                       # Directory for model checkpoints
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Maz2r/PROFiT-Net.git
   cd PROFiT-Net
   ```

2. Set up a Conda environment:
   - Create a new Conda environment (recommended name: `PROFiT-Net`):
     ```bash
     conda create --name PROFiT-Net python=3.8
     conda activate PROFiT-Net
     ```
   - Install required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. Set up the project structure:
   - Ensure `src/` and its subdirectories are in place.
   - Create `data/`, `dataset/`, and `callback/` directories if they don't already exist.

## Target Labels and Abbreviations
To simplify usage, the target properties are abbreviated as follows:
- `exp_bg` → `exp_band_gap2`
- `exp_fe` → `exp_formation_enthalpy`
- `hse06` → `hse06_band_gap`
- `pbe_u` → `pbe_+u_band_gap`

Use these abbreviations in commands when specifying target labels.

## Usage
### Data Preprocessing
Preprocess `.tar` files into `.csv` and `.npy` formats:
```bash
python src/data/preprocess_data.py path_to_tar_file.tar 0.6 0.2 0.2
```

### Train
Train a model for a specific target label (e.g., `exp_bg` for `exp_band_gap2`):
```bash
python src/train/keras/train.py exp_bg
```

### Predict
Make predictions using a pre-trained model:
```bash
python src/predict/keras/predict.py exp_bg
```

## Forked Repository
This project is forked from [PROFiT-Net](https://github.com/sejunkim6370/PROFiT-Net). 
