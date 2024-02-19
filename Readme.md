# Scene Recognition

> _**Rishabh Shah, Julia Yang | Spring '24 | Duke AIPI 590 Applied Computer Vision GP 1**_

This repository contains the code and resources for a Scene Recognition project developed as part of the Duke AIPI 590 Applied Computer Vision GP 1 course. The goal of this project is to develop a system that can accurately identify and classify scenes in images.

### Brainstorm
Explore the initial brainstorming ideas and project planning in our [Google Docs](https://docs.google.com/document/d/12n1ngQg6Nj7vv5hHjfODlDoo-6Alx9YLy7klJtDVxzg/edit).

### Dataset
Download the dataset from [Google Drive](https://drive.google.com/file/d/112pPeJoWmyWjEvpB-AoDWpGp7UGS3QQf/view?usp=sharing).

### Prerequisites

Ensure you have the following dependencies and requirements installed to run the project:

- Python 3.11.0+
- Pytorch 2.20
- Torchvision 0.17.0

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/rishabhshah13/Scene-Recognition.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the project:

1. Ensure you have Python and required dependencies installed.
2. Follow the instructions in the `Dataset` section to obtain the dataset.
3. Train the model using the instructions provided in the `Training` section.
4. Run the UI using the streamlit command

## Dataset

The dataset used in this project is [describe the dataset here, e.g., SceneNet, Places365, etc.]. Due to licensing restrictions, we cannot include the dataset directly in this repository. Please refer to the official website of the dataset to obtain it.

## Model


### Download Data set (without best models)
```bash
python data/download.py --download_data True --download_models False
```

### Download Data set (without best models)
```bash
python data/download.py --download_data True --download_models True
```


### Transform Data set

Data Transformation is done by Albumentations library

### Train Model
```bash
# To Train ResNet18
python models/main.py --model_base resnet18 --num_epochs  100 --batch_size  32 --learning_rate  0.001 --random_seed  42 --use_split --save_checkpoints 1,3 --use_albumentations True --opt adam

# To Train EfficientNet
python models/main.py --model_base enet_s --num_epochs  50 --batch_size  32 --learning_rate  0.001 --random_seed  42 --use_split --save_checkpoints  1,2,3 --use_albumentations True --opt adam

# To Train VGG
python models/main.py --model_base vgg --num_epochs  50 --batch_size  32 --learning_rate  0.001 --random_seed  42 --use_split --save_checkpoints  1,2,3 --use_albumentations False --opt adam

# To Train DenseNet
python models/main.py --model_base densenet --num_epochs  5 --batch_size  32 --learning_rate  0.001 --random_seed  42 --use_split --save_checkpoints 1,3 --use_albumentations True --opt sgd
```

### Run UI
```bash
streamlit run app/UI.py
```

This README provides a comprehensive guide to setting up and using the Scene Recognition project. For further details or assistance, feel free to reach out.

