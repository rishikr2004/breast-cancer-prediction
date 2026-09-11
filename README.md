# 🩺 Breast Cancer Prediction — Neural Network (Keras/TensorFlow) + Streamlit

A binary classifier that predicts whether a breast tumor is **benign** or
**malignant** from 30 numeric cell-nuclei features, using a small
fully-connected neural network (Keras/TensorFlow), deployed as an
interactive **Streamlit** web app.

**🔗 Live demo:** _add your Streamlit Cloud link here after deploying_

## Repository structure
```
.
├── app.py                     # Streamlit app
├── breast_cancer_data.csv     # dataset
├── breast_cancer_model.keras  # trained model
├── scaler.joblib              # fitted StandardScaler
├── feature_names.json         # exact feature column order
├── requirements.txt
└── notebook/                  # original training notebook
```

## Run locally
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to share.streamlit.io → sign in with GitHub.
3. "Create app" → pick this repo → branch `main` → main file `app.py` → Deploy.

## Disclaimer
Educational / portfolio project only — not a medical device.
