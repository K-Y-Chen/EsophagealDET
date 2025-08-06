# **ECD-Dataset Analysis and Detection Model**

## **Overview**
This repository contains the implementation of a detection model based on the ECD-Dataset for esophageal cancer detection.

The source codes of EsophagealDet are placed in ./EsophagealDet

The source codes of other models are placed in ./others

Run huaxi_train.sh or train.sh to start your training.

## **Requirements**
### **Hardware**
- **GPU**: NVIDIA GPU with CUDA support (e.g., NVIDIA RTX 3090 or higher).

### **Software**
- **Python**: 3.8.10.
- **Libraries**:
  Please refer to ./EsophagealDet/EsophagealDet/requirements.txt

Install dependencies using:
```bash
pip install -r requirements.txt

## **Dataset**
### **ECD-Dataset**
The ECD-Dataset is available from the corresponding authors (Yuan or Chen) upon reasonable request.

### **Kvasir-Dataset**
The Kvasir-Dataset is publicly available [here](link-to-dataset). Ensure the dataset is split following [here](link-to-dataset).

## **Training**
Train the model using the command in ./EsophagealDet/EsophagealDet/train_huaxi:
```bash
bash

## **Evaluation**
Evaluate the model performance  using the command in ./EsophagealDet/EsophagealDet/test_huaxi:
```bash
bash

## **Detection**
Perform inference on new images using the command in ./EsophagealDet/EsophagealDet/train_huaxi:
```bash
bash
