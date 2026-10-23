"""
Train a used-car price prediction model.
Note: The bundled dataset is SYNTHETIC demo data, created to make this project runnable.
For a serious evaluation, replace it with a real, properly licensed used-car dataset.
"""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "car_data.csv"
MODEL_PATH = ROOT / "car_price_model.joblib"

df = pd.read_csv(DATA_PATH)
target = "selling_price_lakh"
features = ["brand", "year", "km_driven", "fuel", "transmission", "owner"]
X = df[features]
y = df[target]

categorical = ["brand", "fuel", "transmission"]
numeric = ["year", "km_driven", "owner"]

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
        ("numeric", SimpleImputer(strategy="median"), numeric),
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=120, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

joblib.dump(model, MODEL_PATH)
print("Model trained and saved to:", MODEL_PATH)
print(f"Test MAE: {mae:.2f} lakh rupees")
print(f"Test R² score: {r2:.3f}")
print("Reminder: these metrics reflect synthetic demo data, not real-market accuracy.")
