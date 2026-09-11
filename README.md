# 🩺 Breast Cancer Prediction using Deep Learning

An interactive web application that predicts whether a breast tumour is likely **benign** or **malignant** using a neural-network model built with TensorFlow and Keras.

> **Important:** This is an educational machine-learning project only. It is not a medical device and must not be used for medical diagnosis, treatment, or clinical decision-making.

## Overview

This project uses the Breast Cancer Wisconsin (Diagnostic) dataset, which contains 30 numeric measurements computed from digitized images of fine needle aspirate (FNA) samples of breast masses.

The trained model receives these measurements, applies the same feature scaling used during training, and predicts the probability that the tumour belongs to the malignant class.

## Features

- Interactive manual prediction using 30 input sliders
- Pre-filled feature values based on dataset averages
- Example benign-like and malignant-like inputs
- Batch prediction by uploading a CSV file
- Downloadable CSV prediction results
- Saved Keras model and scikit-learn scaler for consistent predictions
- Educational disclaimer within the application

## Model Architecture

```text
Input layer: 30 features
        ↓
Dense layer: 16 neurons, ReLU activation
        ↓
Dense layer: 8 neurons, ReLU activation
        ↓
Output layer: 1 neuron, Sigmoid activation
```

The sigmoid output represents the predicted probability of the malignant class.

## Machine-Learning Pipeline

```text
Load dataset
    ↓
Remove unnecessary columns
    ↓
Convert labels: Benign = 0, Malignant = 1
    ↓
Split data into training, validation, and test sets
    ↓
Apply StandardScaler
    ↓
Train TensorFlow/Keras neural network
    ↓
Evaluate and save model + scaler
    ↓
Deploy with Streamlit
```

## Model Performance

- Test accuracy: **96.5%**
- Evaluation includes a confusion matrix, classification report, ROC curve, and AUC analysis in the training notebook.

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- scikit-learn
- pandas
- NumPy
- joblib

## Repository Structure

```text
.
├── app.py                         # Streamlit web application
├── breast_cancer_data.csv         # Reference dataset used by the app
├── breast_cancer_model.keras      # Saved trained neural-network model
├── scaler.joblib                  # Saved StandardScaler
├── feature_names.json             # Required feature names and exact order
├── brest_cancer_dl_project.ipynb  # Original model-training notebook
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

## Run Locally

1. Clone or download this repository.

2. Open a terminal in the project folder.

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

5. Open the local URL shown in your terminal, usually:

```text
http://localhost:8501
```

## Using the App

### Manual Input

1. Open the **Manual Input** tab.
2. Adjust the 30 tumour-measurement sliders.
3. Click **Predict**.
4. View the predicted class and probability of malignancy.

### Batch Prediction

1. Open the **Batch Predict (CSV)** tab.
2. Upload a CSV file containing the same 30 feature columns used during training.
3. The app will generate a prediction for every row.
4. Download the results as a CSV file.

The required feature names and their order are stored in `feature_names.json`.

## Deployment

This project is deployed using Streamlit Community Cloud.

- GitHub Repository: https://github.com/rishikr2004/breast-cancer-prediction
- Live Application: Add your Streamlit deployment link here after deployment.

## Disclaimer

This application is created exclusively for educational and portfolio purposes. Its predictions must not be treated as medical advice or a substitute for a qualified healthcare professional.
