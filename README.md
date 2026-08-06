# Housing Price Prediction API

A REST API that predicts median house values in California using a trained
machine learning model, built with **FastAPI** and **scikit-learn**.

This project demonstrates the full path from a trained model to a deployable
service — the core skill gap between "building a model" and being an ML
engineer.

## How It Works

1. A `RandomForestRegressor` is trained on the California Housing dataset.
2. The trained model is serialized to disk using `joblib`.
3. **FastAPI** wraps the model in a REST endpoint, validating incoming
   requests with **Pydantic**.
4. **Uvicorn** serves the API, with interactive docs auto-generated at `/docs`.

## Project Structure
```bash
housing-price-prediction-api/
├── train.py # Trains and saves the model
├── main.py # FastAPI app that serves predictions
├── assets #api working screesnshots
├── requirements.txt # Python dependencies
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/kritikaamohan/housing-price-prediction-api.git
cd housing-price-prediction-api

# Install dependencies
pip install -r requirements.txt
```

## Usage

**1. Train the model** (creates `california_housing_model.joblib`):
```bash
python train.py
```

**2. Run the API:**
```bash
uvicorn main:app --reload
```

**3. Open the interactive docs:**
http://127.0.0.1:8000/docs

## Example Request

`POST /predict`
```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322.0
}
```

**Response:**
```json
{
  "predicted_median_house_value": 4.526
}
```

## Tech Stack

- Python
- scikit-learn (RandomForestRegressor)
- FastAPI
- Uvicorn
- Pydantic
- joblib

## Notes

The model is trained on the classic California Housing dataset (1990 U.S.
Census data) and uses five features: `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, and `Population`.
