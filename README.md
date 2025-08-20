# -OPTIMIZATION-MODEL

Company: CODTECH IT SOLUTIONS

NAME: Kashish Rathi

INTERN ID:CT06DH216G

DOMAIN:Data Science

DURACTION: 6 WEEKS

MENTOR: NEELA SANTOSH

### 📝 Script Description: FastAPI Web Application for California House Price Prediction

This Python script creates a **machine learning-powered web API** using **FastAPI**. It is designed to predict **California housing prices** based on input features, using a trained **Random Forest Regressor** model from `scikit-learn`.

---

### 🔹 1. Dataset Loading and Preparation

The script uses the **California Housing dataset** provided by `scikit-learn`. This dataset includes various features (e.g., median income, house age, number of rooms, population, etc.) and a target variable representing the median house value in various districts in California.

```python
housing = fetch_california_housing()
```

The features and target variable are split into training and testing sets using `train_test_split()`.

---

### 🔹 2. Model Training

A **Random Forest Regressor** is trained on the training data:

```python
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

Random Forest is chosen for its ability to handle non-linear relationships and reduce overfitting through ensemble learning.

---

### 🔹 3. API Input Schema with Pydantic

The `HouseFeatures` class defines the expected format of the input JSON using **Pydantic**, which ensures automatic validation of data sent to the API. This includes fields such as:

* `MedInc`: Median income
* `HouseAge`: Median house age
* `AveRooms`: Average rooms per household
* `Latitude`, `Longitude`, etc.

---

### 🔹 4. FastAPI App Definition

The FastAPI application includes two endpoints:

* `GET /` – A simple welcome route
* `POST /predict` – Accepts a JSON payload of house features, converts it to a DataFrame, and returns the predicted price using the trained model.

Example request to `/predict`:

```json
{
  "MedInc": 8.3,
  "HouseAge": 41,
  "AveRooms": 6.0,
  "AveBedrms": 1.0,
  "Population": 1000,
  "AveOccup": 3.5,
  "Latitude": 34.2,
  "Longitude": -118.3
}
```

---

### 🔹 5. Running the App

The script uses `uvicorn` to run the FastAPI app locally on port `8000` with hot-reloading enabled:

```bash
uvicorn app:app --reload
```

---

#OUTPUT
### ✅ Conclusion

