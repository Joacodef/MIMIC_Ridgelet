# MIMIC-CXR Pathology Classification with Wavelet & Ridgelet Preprocessing

This project provides a flexible deep learning pipeline for the binary classification of various pathologies (e.g., **Cardiomegaly**, **Edema**, **Fracture**) in chest X-ray images from the **MIMIC-CXR-JPG dataset**. The primary scientific goal is to investigate the effectiveness of wavelet-based transforms, such as the **Ridgelet Transform** and **Wavelet Transform**, as preprocessing techniques to enhance feature extraction and improve classification performance.

The current implementation provides a complete and configurable pipeline for training and evaluating models on any chosen pathology from the dataset.

## Features

  * **Configurable Classification Target**: Easily switch the pathology for classification (e.g., from "Fracture" to "Cardiomegaly") by changing a single line in a configuration file.
  * **End-to-End Pipeline**: A full training and validation pipeline for image classification.
  * **Advanced Preprocessing**: Includes customizable `WaveletTransformd` and `RidgeletTransformd` to apply advanced signal processing techniques as part of the data pipeline.
  * **Adaptable Model**: Utilizes pre-trained models (e.g., ResNet, DenseNet) adapted for single-channel or multi-channel medical images.
  * **Automated Data Preparation**: Includes a script to automatically generate balanced training, validation, and test splits based on the configured pathology.
  * **Configuration Driven**: Leverages YAML files for easy management of hyperparameters, model selection, and data paths, ensuring a reproducible workflow.
  * **Best Practices**: Implements on-the-fly data augmentation, model checkpointing, early stopping, and Weights & Biases integration.

## Project Structure

The repository is organized to maintain a clear separation of concerns:

```
MIMIC_Ridgelet/
├── configs/              # YAML configuration files for experiments
├── data/                 # Data storage (e.g., for generated CSV splits)
├── notebooks/            # Jupyter notebooks for exploration and visualization
├── scripts/              # Standalone scripts (e.g., data preparation)
├── src/                  # Main source code
│   ├── config/           # Configuration loading logic
│   ├── data/             # PyTorch Dataset and transform definitions
│   ├── models/           # Model architectures
│   ├── ridgelet/         # Ridgelet transform implementation
│   └── training/         # Training and validation scripts
├── .env.example          # Example environment variables file
└── README.md
```

## Setup and Installation

### 1\. Prerequisites

  * Python 3.8+
  * Conda/Mamba
  * Access to the [MIMIC-CXR-JPG dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). You must have the required credentials and have downloaded the data.

### 2\. Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/joacodef/mimic_ridgelet.git
    cd mimic_ridgelet
    ```

2.  **Create and activate a Conda environment:**
    The `environment.yml` file contains all the necessary dependencies.

    ```bash
    conda env create -f environment.yml
    conda activate magister
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the project root by copying the example file:

    ```bash
    cp .env.example .env
    ```

    Now, edit the `.env` file with the correct **absolute paths** for your system:

      * `MIMIC_CXR_METADATA_PATH`: Path to the directory containing the MIMIC-CXR metadata CSV files.
      * `MIMIC_CXR_P_FOLDERS_PATH`: Path to the `files` directory which contains the patient folders (`p10`, `p11`, etc.).
      * `PROJECT_DATA_FOLDER_PATH`: Path to the `data` directory within this project, where splits will be stored.
      * `PROJECT_OUTPUT_FOLDER_PATH`: Path to a directory where trained models and results will be saved.

## Usage

The entire workflow is controlled by the configuration file. The process is simple: **1. Edit Config -\> 2. Create Splits -\> 3. Train Model**.

### Step 1: Configure Your Experiment

Open a configuration file (e.g., `configs/config_example.yaml`) to define your experiment. The two most important parameters are:

  * `pathology`: The target label for classification (e.g., "Cardiomegaly", "Edema", "Fracture"). This must match a column name in the `mimic-cxr-2.0.0-chexpert.csv.gz` file.
  * `data.train_size`: The desired total size of the balanced training set.

Example for a **Cardiomegaly** experiment with **80,000** training images:

```yaml
# In configs/config_example.yaml
pathology: "Cardiomegaly"
data:
  train_size: 80000
  image_size: 512
# ... other parameters
```

### Step 2: Create Data Splits

Run the `create_splits.py` script, pointing it to your chosen config file. The script will read the `pathology` and `train_size` and automatically generate the corresponding `train.csv`, `validation.csv`, and `test.csv` files.

```bash
python scripts/create_splits.py --config configs/config_example.yaml
```

This will create a new folder, for instance, `data/splits/split_Cardiomegaly_80000/`.

### Step 3: Run Training

Start the training process by pointing the training script to the **same configuration file**. The script will automatically find the correct data splits and save all outputs to a unique, timestamped directory.

```bash
python -m src.training.train --config configs/config_example.yaml
```

The script will log progress to the console and to Weights & Biases (if enabled). Trained models will be saved in the directory specified by `PROJECT_OUTPUT_FOLDER_PATH`, organized by pathology.
