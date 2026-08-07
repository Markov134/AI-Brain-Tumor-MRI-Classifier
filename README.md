# Brain Tumor MRI Classification Using ResNet-18

---

## Project Overview

This project uses deep learning to classify brain MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary

The goal of this project was to build an end-to-end machine learning pipeline that includes data storage, exploratory analysis, image preprocessing, model training, evaluation, and prediction.

---

## Dataset

The project uses the Brain Tumor MRI Dataset containing 7,200 MRI images across four diagnostic categories.

The dataset was organized into:
- Training images
- Testing images

Each image was stored and managed using a SQLite database before being processed for deep learning.

---

## Technologies Used

### Programming
- Python

### Data Processing
- Pandas
- NumPy
- SQLite

### Deep Learning
- PyTorch
- Torchvision
- ResNet-18

### Visualization
- Matplotlib
- Power BI

### Machine Learning Evaluation
- Scikit-learn

---

## Project Workflow

1. Dataset Collection
   - Loaded MRI images from the dataset.

2. Database Creation
   - Stored image metadata in SQLite.
   - Included filename, filepath, diagnosis, image dimensions, and dataset split.

3. Exploratory Data Analysis
   - Analyzed dataset structure.
   - Examined class distribution.
   - Visualized sample MRI images.
   - Compared training and testing distribution.

4. Image Preprocessing
   - Resized images to 224x224.
   - Converted images into tensors.
   - Normalized pixel values.

5. Model Training
   - Used transfer learning with ResNet-18.
   - Modified the final layer to classify four MRI categories.

6. Model Evaluation
   - Evaluated performance using:
      - Accuracy
      - Precision
      - Recall
      - F1 Score
      - Confusion Matrix

7. Prediction System
   - Built a prediction pipeline capable of classifying individual MRI images.

8. Power BI Dashboard
   - Created a dashboard to visualize dataset information and model performance.

---

## Model Performance

The ResNet-18 model achieved the following results:

| Metric | Score |
|--------|-------|
| Accuracy | 92.44% |
| Precision | 92.61% |
| Recall | 92.44% |
| F1 Score | 92.38% |

The results demonstrate that the model was able to effectively classify MRI images into their corresponding categories.

---

## Power BI Dashboard

The project includes a Power BI dashboard containing:

- Total MRI image count
- MRI images by diagnosis
- Training/testing split
- Model evaluation metrics
- Confusion matrix visualization
- Example prediction results

---

## Repository Structure

MRI-Brain-Scan-Project\
│\
├── images\
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
├── src
│   ├── database.py
│   ├── preprocessing.py
│   ├── model.py
│   └── predict.py
│
├── dashboard
│   └── MRI_Brain_Tumor_Dashboard.pbix
│
|
├── README.md
└── .gitignore

## Conclusion

This project demonstrates an end-to-end deep learning workflow for brain tumor MRI classification. 
By combining database management, image preprocessing, transfer learning, model evaluation, and visualization, the project provides a complete pipeline from raw MRI images to prediction results.
