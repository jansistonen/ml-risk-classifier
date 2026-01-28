# Machine Learning Risk Classifier (REST API)

This project is a machine learning–based application that classifies breast cancer risk as **malignant** or **benign**.

The model is exposed as a REST API, allowing predictions to be requested over HTTP by sending patient feature data in JSON format.

Live API documentation (Swagger UI):  
https://risk-api-service-1769105645.azurewebsites.net/docs

---

## Project Overview

- Machine learning classification service
- Breast cancer risk prediction
- FastAPI-based REST API
- Dockerized and deployable to cloud platforms (Azure)

---

## Dataset

The model is trained using the **Breast Cancer Wisconsin dataset** provided by the Scikit-learn library.

The dataset contains numerical features describing cell nuclei, such as:

- radius
- texture
- perimeter
- area
- smoothness

These features are used to predict whether a tumor is malignant or benign.

---

## Model Development

Two different machine learning approaches were evaluated during development.

### Logistic Regression (Initial Version)

The first implementation used logistic regression, which:

- learns a linear decision boundary between classes
- outputs probabilities using the sigmoid function

Mathematical formulation:




This approach was simple and interpretable but had limited predictive performance.

---

### Random Forest Classifier (Current Version)

The current production model uses a Random Forest classifier.

Reasons for the change:

- higher accuracy
- robustness to noise
- ability to model non-linear feature interactions

Random Forest combines predictions from multiple decision trees. Each tree votes on the outcome, and the final prediction is based on majority voting.

---

## Project Workflow

The application follows a clear pipeline from data processing to deployment:

1. Data Loading  
   - The breast cancer dataset is loaded from `sklearn.datasets`

2. Model Training (`train.py`)  
   - The model is trained on the dataset  
   - The trained model is saved as a `.joblib` file

3. API Layer (`main.py`)  
   - A FastAPI server loads the trained model  
   - Exposes prediction endpoints via HTTP

4. Containerization  
   - The application is packaged into a Docker image  
   - Ensures consistent behavior locally and in the cloud

---

## Local Development Guide

### Requirements

- Python 3.9+
- pip
- virtualenv (recommended)
- Docker (optional)

---

### Installation

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
# Train the Model

Run the training script to generate the machine learning model:

```bash
python train.py
```
This will create a serialized model file used by the API.

# Run the API server
Start the FastAPI application using Uvicorn:
```bash
uvicorn main:app --reload
```
The service will be available at:

API base URL: http://127.0.0.1:8000

Swagger UI: http://127.0.0.1:8000/docs

# Running with Docker
Build the Docker image:
```bash
docker build -t ml-risk-classifier .
```
Run the container:
```bash
docker run -p 8000:8000 ml-risk-classifier
```
# The API documentation will be accessible at:

http://localhost:8000/docs

# API Usage

The API accepts a JSON payload containing numerical feature values and returns a risk classification.

Detailed request and response schemas can be found in the Swagger UI.




