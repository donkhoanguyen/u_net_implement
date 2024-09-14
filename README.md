# U-Net Object Detection

This repository contains an implementation of the U-Net model for object detection. The model has been trained on the Carvana Image Masking Challenge dataset to achieve high accuracy in segmentation tasks.

## Repository Structure

- `dataset.py`: Script for loading and processing the dataset.
- `model.py`: Implementation of the U-Net model architecture.
- `train.py`: Script for training the U-Net model.
- `predict.py`: Script for running predictions with the trained model.
- `utils.py`: Utility functions used in training and prediction.
- `model_test.ipynb`: Jupyter notebook for testing and evaluating the model.
- `requirements.txt`: List of required Python packages.
- `carvana-image-masking-challenge/`: Directory containing the dataset.
- `prediction/`: Directory where prediction results are saved.
- `saved_images/`: Directory for storing saved images.

## Model Details
Model: U-Net
Accuracy: 98% on the validation set
Data Augmentation: Utilizes Albumentations for data augmentation and transformation.

## Prerequisites

1. Install the required packages by running:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure you have the dataset available in the carvana-image-masking-challenge directory.

## Training
To train the U-Net model, run:

    ```bash
    python train.py
    ```
This will save the trained model checkpoints and intermediate results.

## Prediction
To make predictions on new images, run:

    ```bash
    python predict.py
    ```
The results will be saved in the prediction directory.

## Testing
We obtained a 98% accuracy with a DIce score of 0.97 on 48 validation images.

## License
This project is licensed under the MIT License.

## Acknowledgements
The Carvana Image Masking Challenge for providing the dataset.
alladinpersson for the tutorial
Feel free to open issues or submit pull requests if you have any questions or improvements!