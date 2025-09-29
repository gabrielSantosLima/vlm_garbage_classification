![](./assets/badge.png)

# Using VLMs for Garbage Classification

Comparing the usage of Visual Language Models (VLM) and Convolutional Neural Network to classify garbage images of the dataset ["Garbage Image Dataset"](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset/) by Farzad Nekouei.

Tested models:
| Model Name                         | Params | Accuracy | Precision | Recall | F1-Score |
|------------------------------------|--------|----------|-----------|--------|----------|
| EfficientNetV2B2                   | 8M     | 90.51%   | 92.11%    | 88.18% | 89.62%   |
| SmolVLM-1.7B (Zero-Shot Prompting) | 1.7B   | 79.05%   | 68.57%    | 69.19% | 68.62%   |
| SmolVLM-500M (Zero-Shot Prompting) | 500M   | 62.85%   | 62.25%    | 54.98% | 55.32%   |
| SmolVLM-1.7B (Few-Shot Prompting)  | 1.7B   | 54.55%   | 62.68%    | 50.68% | 48.70%   |
| SmolVLM-500M (Few-Shot Prompting)  | 500M   | 53.36%   | 51.34%    | 50.73% | 45.13%   |
| SmolVLM-256M (Zero-Shot Prompting) | 256M   | 50.20%   | 48.91%    | 43.58% | 44.38%   |
| SmolVLM-256M (Few-Shot Prompting)  | 256M   | 50.20%   | 52.75%    | 46.30% | 43.60%   |

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
- pytorch
- transfomers

> See the complete configuration in [environment.win.yml](./environment.win.yml)

