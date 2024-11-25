# Breast Cancer Prediction Web App

This web app predicts whether a breast tumor is malignant or benign based on input features, using machine learning models such as Multi-layer Perceptron (MLP) and Artificial Neural Networks (ANN).

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)

## Introduction

Breast cancer is one of the leading causes of cancer-related deaths. Early diagnosis and accurate prediction can help doctors to provide timely treatment. This app allows users to input various tumor-related features, and it predicts whether the tumor is benign or malignant.

The app uses a trained machine learning model (ANN) to make predictions based on the input data.

## Requirements

To run this project locally or deploy it, you need to have the following Python libraries:

- `streamlit`
- `tensorflow`
- `scikit-learn`
- `joblib`
- `numpy`
- `pandas`
- `matplotlib`

These libraries can be installed using the `requirements.txt` file.

## Installation



```bash
git clone https://github.com/your_username/breast-cancer-prediction.git
cd breast-cancer-prediction
```
```
python -m venv venv
source venv/bin/activate  # For Windows, use `venv\Scripts\activate`
```

```
pip install -r requirements.txt
```

```
streamlit run app.py
```

