
# 🚀 PROFiT-Net: Optimizing Material Property Prediction

PROFiT-Net is a machine learning model designed to optimize the prediction of material properties by leveraging deep learning architectures. It integrates domain-specific knowledge with advanced neural networks to improve accuracy and efficiency in predicting key material properties such as band gaps, formation enthalpies, and other physical or chemical attributes. The model focuses on feature extraction and training strategies tailored for material science datasets, making it a powerful tool for accelerating research in materials discovery and design.

---

## 🌟 Features
- **Data Preprocessing**:
  - Extracts `.tar` files and processes data into `.csv` and `.npy` formats.
  - Splits data into training, validation, and test sets.

- **Model Training**:
  - Train CNN models for specific target labels using Keras or PyTorch.
  - Supports model checkpointing for resuming training or prediction.

- **Prediction**:
  - Load pre-trained models to make predictions on test data.
  - Provides performance statistics (MAE, MSE, etc.) after predictions.

- **WandB Integration**:
  - Automatically logs metrics, configurations, and model artifacts to Weights and Biases (WandB) for tracking and analysis.

---

## 📂 Data Directory Structure
The data used in this project is organized as follows:
- **`/dataset/{target_name}`**: Contains the original `.tar` and extracted `.csv` files. These are the raw data files.
- **`/data/{target_name}/`**: Contains processed `.npy` files used for training and prediction.

### ⚠️ Usage Note:
To use the data for training or prediction:
1. Place the original `.tar` file under `/dataset/{target_name}`.
2. Use the script `src/data/preprocess_data.py` to process the `.tar` file and generate `.npy` files:
   ```bash
   python src/data/preprocess_data.py path_to_tar_file.tar 0.6 0.2 0.2
   ```
3. The processed `.npy` files will be saved under `/data/{target_name}/`.

---

## 📦 Packages Used
The following major packages and libraries are used in this project:
- **Python** >= 3.8
- **Keras** >= 2.8.0
- **TensorFlow** >= 2.8.0
- **PyTorch** >= 2.0.0
- **scikit-learn** >= 0.24.0
- **Numpy** >= 1.19.0
- **Pandas** >= 1.1.0
- **WandB** >= 0.12.0
- **PyNVML** >= 11.0.0

Refer to the `requirements.txt` or `requirements_torch.yml` for detailed dependencies.

---

## 🛠️ Installation

### Keras Version
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

### PyTorch Version
1. Clone the repository (if not already done):
   ```bash
   git clone https://github.com/Maz2r/PROFiT-Net.git
   cd PROFiT-Net
   ```

2. Set up a Conda environment:
   - Create a new Conda environment (recommended name: `PROFiT-Net_torch`):
     ```bash
     conda env create -f requirements_torch.yml
     conda activate PROFiT-Net_torch
     ```

3. Log into WandB:
   - Use the following command to log into WandB:
     ```bash
     wandb login
     ```
   - Enter your API key when prompted. You can find your API key on your [WandB account page](https://wandb.ai/authorize).

---

## 🏷️ Target Labels and Abbreviations
To simplify usage, the target properties are abbreviated as follows:
- `exp_bg` → `exp_band_gap2`
- `exp_fe` → `exp_formation_enthalpy`
- `hse06` → `hse06_band_gap`
- `pbe_u` → `pbe_+u_band_gap`

Use these abbreviations in commands when specifying target labels.

---

## 📝 Usage

### 📂 Data Preprocessing
Preprocess `.tar` files into `.csv` and `.npy` formats:
```bash
python src/data/preprocess_data.py path_to_tar_file.tar 0.6 0.2 0.2
```

### 🏋️‍♂️ Training
#### Keras Version
Train a model for a specific target label (e.g., `exp_bg` for `exp_band_gap2`):
```bash
python src/train/keras/train.py exp_bg
```

#### PyTorch Version
Train a model for a specific target label (e.g., `exp_bg` for `exp_band_gap2`):
```bash
python src/train/pytorch/train.py exp_bg
```

### 🔍 Prediction
#### Keras Version
Make predictions using a pre-trained model:
```bash
python src/predict/keras/predict.py exp_bg
```

#### PyTorch Version
Make predictions using a pre-trained model:
1. Provide the target abbreviation and the WandB run ID of the desired model checkpoint:
   ```bash
   python src/predict/pytorch/predict.py exp_bg xyz123
   ```
   - `exp_bg` is the target abbreviation.
   - `xyz123` is the run ID associated with the model.

2. The script will:
   - Fetch the weights from WandB and save them under `fetched_weights/`.
   - Make predictions on the test data.
   - Log metrics (e.g., MAE, MSE) and other details to WandB.
   - Print a detailed prediction summary to the console.

---

## 🌐 WandB Integration
### Key Features
- **Model Artifact Logging**:
  - Checkpoints and best weights are logged as artifacts to WandB, enabling easy retrieval and versioning.
  
- **Metrics Logging**:
  - Logs metrics such as loss, MAE, and MSE for training and prediction tasks.

### Retrieving Model Weights
To fetch the model weights stored in WandB as artifacts:
1. Initialize WandB:
   ```python
   import wandb
   wandb.init(project="PROFiT-Net", entity="cnmd-phb-postech")
   ```

2. Fetch the desired artifact:
   ```python
   artifact = wandb.use_artifact('exp_bg_xyz123:latest', type='model')
   artifact_dir = artifact.download()
   print(f"Model weights downloaded to: {artifact_dir}")
   ```

---

## 🔗 Forked Repository
This project is forked from [PROFiT-Net](https://github.com/sejunkim6370/PROFiT-Net). 
