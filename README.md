![](./assets/badge.png)

# Using VLMs for Garbage Classification

> :exclamation: This is a work in progress

Comparing the usage of Visual Language Models (VLM) and Convolutional Neural Network to classify garbage images of the dataset ["Garbage Image Dataset"](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset/) by Farzad Nekouei.

Tested models:
| Model Name | Accuracy | Precision | Recall | F1-Score |
|-------------------------------|----------|-----------|--------|----------|
| EfficientNetV2B2 | 90.51% | 92.11% | 88.18% | 89.62% |
| SmolVLM (Zero-Shot Prompting) | WIP | WIP | WIP | WIP |
| SmolVLM (Few-Shot Prompting) | WIP | WIP | WIP | WIP |

## Requirements

- conda==25.3.1
- python==3.10
- tensorflow
- keras
- pandas
- numpy
- matplotlib
- seaborn
- jupyterlab
- opencv-python
- scikit-learn

> See the complete configuration in [environment.win.yml](./environment.win.yml)
