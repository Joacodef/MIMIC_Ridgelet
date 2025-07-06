# MIMIC-CXR Fracture Detection with Shearlet Preprocessing

This project explores the use of deep learning for the binary classification of fractures in chest X-ray images from the **MIMIC-CXR-JPG dataset**. The primary scientific goal is to investigate the effectiveness of the **Shearlet Transform** as a preprocessing technique to enhance feature extraction and potentially improve classification performance.

The current implementation provides a complete baseline pipeline using a ResNet-based architecture.

## Features

* **End-to-End Pipeline**: A full training and validation pipeline for image classification.
* **Adaptable Model**: Utilizes pre-trained ResNet models (`resnet18`, `resnet34`, `resnet50`) adapted for single-channel medical images.
* **Data Preparation**: Includes a script to generate balanced training, validation, and test splits from the raw MIMIC-CXR metadata.
* **Configuration Driven**: Leverages YAML files for easy management of hyperparameters, model selection, and data paths.
* **Data Augmentation**: Integrates with the `monai` library for on-the-fly data augmentation.
* **Best Practices**: Implements model checkpointing to save the best models and early stopping to prevent overfitting.

## Project Structure

The repository is organized to maintain a clear separation of concerns:

```
MIMIC_Ridgelet/
├── configs/              # YAML configuration files for experiments
├── scripts/              # Standalone scripts (e.g., data preparation)
├── src/                  # Main source code
│   ├── config/           # Configuration loading logic
│   ├── data/             # PyTorch Dataset and DataLoader definitions
│   ├── models/           # Model architectures
│   └── training/         # Training, validation, and evaluation loops
├── tests/                # Unit and integration tests
├── .env.example          # Example environment variables file
└── README.md
```

## Setup and Installation

### 1. Prerequisites

* Python 3.8+
* Access to the [MIMIC-CXR-JPG dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). You must have the required credentials and have downloaded the data.

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/joacodef/mimic_ridgelet.git](https://github.com/joacodef/mimic_ridgelet.git)
    cd MIMIC_Ridgelet
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    *(Note: A `requirements.txt` file should be created for this project. For now, you will need to install libraries like `torch`, `torchvision`, `monai`, `pandas`, `scikit-learn`, `pyyaml`, `python-dotenv`, and `tqdm`.)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a `.env` file in the project root by copying the example file:
    ```bash
    cp .env.example .env
    ```
    Now, edit the `.env` file with the correct absolute paths for your system:
    * `MIMIC_CXR_METADATA_PATH`: Path to the directory containing the MIMIC-CXR metadata CSV files (e.g., `mimic-cxr-2.0.0-metadata.csv.gz`).
    * `MIMIC_CXR_P_FOLDERS_PATH`: Path to the `files` directory which contains the patient folders (`p10`, `p11`, etc.).
    * `PROJECT_DATA_FOLDER_PATH`: Path to the `data` directory within this project, where splits will be stored.
    * `PROJECT_OUTPUT_FOLDER_PATH`: Path to a directory where trained models and results will be saved.

## Usage

### Step 1: Create Data Splits

Before training, you need to generate the CSV files that define your dataset splits. Run the `create_splits.py` script. The `--train_size` argument specifies the total number of images in the balanced training set (half with fractures, half without).

```bash
python scripts/create_splits.py --train_size 2000 --val_size 400 --test_size 400
```

This will create a new folder `data/splits/split_2000` containing `train.csv`, `validation.csv`, and `test.csv`.

### Step 2: Configure the Training Run

Open `configs/config.yaml` to adjust parameters for your experiment. Key settings include:

* `data.split_folder_name`: Must match the folder created in Step 1 (e.g., `split_2000`).
* `data.image_size`: The size to which images will be resized.
* `model.base_model`: The ResNet architecture to use (`resnet18`, `resnet34`, or `resnet50`).
* `training.epochs`, `training.learning_rate`, etc.

### Step 3: Run Training

Start the training process by pointing to your configuration file. Using `-m` ensures correct module resolution from the `src` directory.

```bash
python -m src.training.train --config configs/config.yaml
```

The script will log progress to the console. Trained models and the configuration file for the run will be saved in the directory specified by `PROJECT_OUTPUT_FOLDER_PATH`.

## Roadmap

The following features and experiments are planned for future development:

* [ ] **Ridgelet Transform Integration**: Implement the Shearlet transform as a preprocessing layer applied before the model's first convolutional layer.
* [ ] **Comparative Analysis**: Rigorously evaluate and compare model performance (AUC, F1-score, Precision, Recall) with and without Shearlet preprocessing.
* [ ] **Advanced Models**: Experiment with other modern CNN architectures.
* [ ] **Inference Script**: Create a script to run inference on new, unseen images using a saved model checkpoint.
