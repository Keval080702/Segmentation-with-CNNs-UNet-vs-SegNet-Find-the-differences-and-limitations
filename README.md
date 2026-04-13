# Semantic Image Segmentation with UNet and SegNet

This project implements and evaluates two popular deep learning architectures, **UNet** and **SegNet**, for semantic image segmentation. The core objective is to accurately classify each pixel in an image, assigning it a specific class label. The models are trained and tested on two distinct datasets: the **ISIC 2018 Skin Lesion** dataset for medical image segmentation and the **Cityscapes dataset** for urban scene understanding.p

## 1. Project Overview

Semantic segmentation is a fundamental task in computer vision that involves dense prediction, where every pixel in an image is classified into a specific object category. This repository provides a comprehensive framework for setting up, training, evaluating, and visualizing the performance of UNet and SegNet models.

## 2. Key Features

* **Multiple Architectures:**
    * **UNet:** A widely used encoder-decoder architecture known for its skip connections, enabling precise localization.
    * **SegNet:** Another encoder-decoder architecture that uses max-pooling indices for efficient upsampling in the decoder.
* **Diverse Encoders:**
    * For UNet, various pre-trained encoders from `segmentation_models_pytorch` are supported, including `VGG16`, `ResNet34`, and `EfficientNetB3`, leveraging transfer learning.
    * SegNet utilizes a custom VGG-like encoder architecture, trained from scratch.
* **Flexible Loss Functions:**
    * **Cross-Entropy Loss:** A standard pixel-wise classification loss, with support for class weighting to address imbalance.
    * **Dice Loss:** Directly optimizes for the overlap between predicted and true masks, beneficial for imbalanced classes and boundary accuracy.
    * **Combined Cross-Entropy and Dice Loss:** A weighted sum of both, aiming to leverage their respective strengths.
* **Robust Data Augmentation:** Utilizes the `Albumentations` library for on-the-fly data augmentation, including:
    * Geometric transforms (HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, OpticalDistortion, GridDistortion).
    * Color augmentations (ColorJitter, RandomBrightnessContrast, GaussNoise, CLAHE).
    * Structural augmentations (CoarseDropout).
* **Comprehensive Training Pipeline:**
    * Adam optimizer with an initial learning rate of `0.0001` and `weight_decay` of `1e-4`.
    * `ReduceLROnPlateau` scheduler to dynamically adjust the learning rate based on validation mIoU.
    * Support for Automatic Mixed Precision (AMP) for faster training on CUDA-enabled GPUs.
* **Evaluation Metrics:** Model performance is rigorously assessed using:
    * **Pixel Accuracy:** Percentage of correctly classified pixels.
    * **Mean Intersection over Union (mIoU):** The primary metric, providing a robust measure of segmentation quality by averaging IoU scores across all classes.
* **Experiment Management:**
    * Automated saving of best models based on validation mIoU.
    * Generation of plots for training and validation metrics (loss, accuracy, mIoU) over epochs.
    * Summarization of all experiment results (configurations, best mIoU, status, errors) into CSV files for easy analysis.
    * Visualization of sample predictions on the test set for qualitative assessment.

## 3. Datasets

This project is designed to work with the following datasets. **Please note that these datasets must be downloaded separately and placed in the specified directory structure.** The notebooks currently assume a base directory of `/home/5177/Deep_Vision/`.

* **ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection**
    * **Purpose:** Binary segmentation (lesion vs. background) of dermoscopic images.
    * **Expected Directory Structure (relative to `ISIC_BASE_DIR`):**
        ```
        ISIC2018_Task1-2_Training_Input/
        ISIC2018_Task1_Training_GroundTruth/ISIC2018_Task1_Training_GroundTruth/
        ISIC2018_Task1-2_Test_Input/
        ISIC2018_Task1_Test_GroundTruth/ISIC2018_Task1_Test_GroundTruth/
        ```

* **Cityscapes Dataset**
    * **Purpose:** Multi-class semantic segmentation of urban street scenes (19 semantic classes).
    * **Expected Directory Structure (relative to `CITYSCAPES_ROOT_DATA_DIR`):**
        ```
        leftImg8bit_trainvaltest/leftImg8bit/train/<city>/
        gtFine_trainvaltest/gtFine/train/<city>/
        leftImg8bit_trainvaltest/leftImg8bit/val/<city>/
        gtFine_trainvaltest/gtFine/val/<city>/
        ```
    * **Note:** The `_gtFine_labelIds.png` masks are used for ground truth.

## 4. Installation

To set up your environment and run the project, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```

3.  **Install PyTorch:**
    Install PyTorch with CUDA support if you have an NVIDIA GPU for accelerated training. Refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for the most up-to-date installation instructions based on your system and CUDA version.
    * Example for CUDA 11.8:
        ```bash
        pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
        ```
    * For CPU-only:
        ```bash
        pip install torch torchvision torchaudio
        ```

4.  **Install Other Dependencies:**
    ```bash
    pip install segmentation-models-pytorch albumentations opencv-python matplotlib pandas tqdm numpy
    ```

## 5. Usage

This project is organized into multiple Jupyter Notebooks, each focusing on specific architecture-dataset combinations. To run the experiments, open these notebooks in a Jupyter environment and execute their cells sequentially.

1.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Your web browser will open, displaying the Jupyter interface.

2.  **Navigate to Project Files:**
    Open the desired notebook file from the list below.

3.  **Configure Dataset Paths:**
    In the first code cell of each notebook, **verify and update the `ISIC_BASE_DIR` and `CITYSCAPES_ROOT_DATA_DIR` variables** to point to the root directories where you have downloaded and extracted your datasets.

4.  **Execute Notebooks:**
    Run all cells in the chosen notebook sequentially. The notebooks are designed to be self-contained for their specific task.

### Project Notebooks:

* `Unet_ISIC2018.ipynb`:
    * **Purpose:** Trains and evaluates UNet models on the ISIC 2018 dataset.
    * **Key Sections:** Initial setup, ISIC Dataset definition, core utilities (metrics, dataloaders, model/loss functions), training/validation loops, UNet experiment configuration, training orchestration, and test set evaluation with visualizations.

* `Segnet_ISIC2018.ipynb`:
    * **Purpose:** Trains and evaluates SegNet models on the ISIC 2018 dataset.
    * **Key Sections:** Initial setup, ISIC Dataset definition (with enhanced augmentations), core utilities (metrics, plotting, class weights, dataloaders), custom SegNet model definition, model/loss function utilities, training/evaluation loops, SegNet experiment configuration, training orchestration, and test set evaluation with visualizations.

* `Unet_Cityscrep1.ipynb`:
    * **Purpose:** Trains and evaluates UNet models on the Cityscapes dataset.
    * **Key Sections:** Setup and configuration for Cityscapes, Cityscapes Dataset class and transforms, general utilities (metrics, plotting, class weights, dataloaders, model/loss functions), training/validation loop functions, UNet experiment configurations, and training orchestration.

* `Segnet_Cityscrep.ipynb`:
    * **Purpose:** Trains and evaluates SegNet models on the Cityscapes dataset.
    * **Key Sections:** Setup and configuration for Cityscapes, Cityscapes Dataset class and transforms (enhanced), custom SegNet model definition, general utilities (metrics, plotting, class weights, dataloaders, model/loss functions), training/evaluation loop functions, SegNet experiment configurations, and training orchestration.

* `Unet_and_Segnet_both_Cityscrep.ipynb`:
    * **Purpose:** This notebook appears to be a consolidated version for Cityscapes, potentially allowing for training and evaluation of both UNet and SegNet within a single flow. It includes shared Cityscapes setup, UNet model definition, loss/metrics, and training/validation functions.
    * **Note:** If you intend to run both UNet and SegNet experiments on Cityscapes, this notebook might offer a streamlined approach.


## 6. Accessing Trained Models

Trained models and other project artifacts can be accessed via our GitLab repository:
<https://git.oth-aw.de/5177/segmentation-with-cnns-unet-vs-segnet-find-the-differences-and-limitations>
<ttps://github.com/Keval080702/Segmentation-with-CNNs-UNet-vs-SegNet-Find-the-differences-and-limitations>
