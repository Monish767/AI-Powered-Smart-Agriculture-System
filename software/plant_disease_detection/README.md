# Plant Disease Detection Module

This module implements the AI component of the
**AI-Powered Smart Agriculture System**.

## Description
The system uses a Convolutional Neural Network (CNN) based on
**MobileNetV2** to classify plant leaf images as healthy or diseased.

## Components

- **data/**  
  Dataset structure, labels, and data preparation instructions.

- **notebooks/**  
  Jupyter notebooks used for data exploration and experimentation.

- **model/**  
  Model architecture, training scripts, and saved model files.

- **inference/**  
  Scripts for loading the trained model and performing predictions.

- **streamlit_app/**  
  Streamlit-based web interface for uploading images and viewing predictions.

## Notes
- The dataset itself is not stored in the repository due to size constraints.
- The model is trained using transfer learning on MobileNetV2.