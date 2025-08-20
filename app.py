# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import uvicorn

# 1. Load and prepare the California housing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Define input data schema
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# 5. Initialize FastAPI
app = FastAPI(title="🏠 California House Price Prediction API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the California House Price Prediction API"}

@app.post("/predict")
def predict(features: HouseFeatures):
    input_data = pd.DataFrame([features.dict()])
    prediction = model.predict(input_data)[0]
    return {"predicted_price": prediction}

# 6. Run the server
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
