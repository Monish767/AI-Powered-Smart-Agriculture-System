# Dataset Information

This project uses the **PlantVillage dataset** for training and evaluating
the plant disease detection model.

## Dataset Name
**PlantVillage Dataset**

## Source
- Publicly available plant disease image dataset
- Widely used for benchmarking plant disease classification models

## Description
The PlantVillage dataset contains labeled images of healthy and diseased
plant leaves captured under controlled conditions.  
Each image belongs to a specific crop–disease class.

In this project, the dataset is used to train a **CNN-based MobileNetV2**
model for image-based plant disease classification.

## Classes
The dataset includes multiple crop and disease categories such as:
- Healthy leaves
- Bacterial diseases
- Fungal diseases
- Viral diseases

> The exact class names used for training are listed in  
> `labels/PlantCareAI_Labels.txt`.

## Dataset Availability
Due to size constraints, the PlantVillage dataset is **not included**
in this repository.

To reproduce the results:
1. Download the PlantVillage dataset from a public source
2. Update the dataset path in the training scripts