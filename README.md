# Brain Tumor MRI Classification Using ResNet-18

Deep Learning + Computer Vision + MRI Image Classification + AI Model Development + SQL

This project uses the **Brain Tumor MRI Dataset**, sourced from Kaggle. 

## 🧠 Project Overview

This project explores and develops an AI-based system for classifying brain MRI images into different tumor categories. The objective is to build a deep learning model capable of identifying patterns within MRI scans and accurately predicting whether an image belongs to one of four categories: Glioma, Meningioma, No Tumor, or Pituitary.

The project follows a complete deep learning workflow:

- Dataset organization and database creation using SQLite
- Exploratory Data Analysis (EDA) and image visualization
- Image preprocessing and transformation
- Deep learning model development using ResNet-18
- Model training and evaluation
- Classification metrics analysis and confusion matrix evaluation
- Individual MRI image prediction with confidence scores
- Power BI dashboard visualization for model results

## 📊 Dataset

The dataset contains the following columns:

| Feature | Description |
| ---------------- | ------------------------------------------------------------------ |
| Image | MRI scan image |
| Diagnosis | Tumor category for each MRI image |
| Split | Dataset training or testing set |
| Image Dimensions | Original width and height of each MRI image before preprocessing |
| File Size | Storage size of each image file |
| File Path | Location of the MRI image used for loading and processing |

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet-18 (Transfer Learning)
- Pandas
- SQLite
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Power BI

## 📈 Model Performance

The ResNet-18 model achieved the following results:

| Metric | Score |
|--------|-------|
| Accuracy | 92.44% |
| Precision | 92.61% |
| Recall | 92.44% |
| F1 Score | 92.38% |

## 📊 Power BI Dashboard

The project includes a Power BI dashboard containing:

- Total MRI image count
- MRI images by diagnosis
- Training/testing split
- Model evaluation metrics
- Confusion matrix visualization
- Example prediction results


## 📁 Project Structure

```
MRI-Brain-Scan-Project
│
├── images
│   ├── confusion_matrix.png
│   ├── training_loss.png
│   └── sample_mri_images.png
│
├── notebooks
│   ├── 01_database_creation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing_and_training.ipynb
│   └── 04_model_results.ipynb
│
├── src\
│   ├── database.py\
│   ├── preprocessing.py\
│   ├── model.py\
│   └── predict.py\
│\
├── dashboard\
│   └── MRI_Brain_Tumor_Dashboard.pbix\
│\
|\
├── README.md\
└── .gitignore
```
