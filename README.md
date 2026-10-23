# Used Car Price Prediction Website

A beginner-friendly Machine Learning mini project using Python, Pandas, Scikit-learn, Joblib, and Streamlit.

## Important dataset note
The included `dataset/car_data.csv` is **synthetic demo data**, generated only so the project can run immediately. It is not collected from real car sales and must not be presented as a real market dataset or as evidence of real-world accuracy. For a stronger academic submission, replace it with a real, properly licensed dataset such as a suitable CarDekho dataset from Kaggle and update the CSV columns as needed.

## Run on Windows

1. Install Python 3.10 or newer from https://www.python.org/downloads/
2. Extract this folder to your computer.
3. Open the folder in VS Code.
4. In VS Code, choose Terminal → New Terminal.
5. Create a virtual environment (recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If activation is blocked, you can skip activation and run the install commands with `python -m pip`.

6. Install packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

7. Train and save the model:

   ```powershell
   python train_model.py
   ```

8. Start the website:

   ```powershell
   python -m streamlit run app.py
   ```

9. A browser tab should open. If not, visit http://localhost:8501

## Files
- `app.py`: website interface and prediction
- `train_model.py`: trains and evaluates the model
- `dataset/car_data.csv`: synthetic demo dataset
- `requirements.txt`: required packages

## ML workflow
1. Read the CSV data.
2. Separate input features from the target price.
3. Encode categorical features and impute missing values.
4. Split the data into training and testing subsets.
5. Train a Random Forest Regressor.
6. Evaluate using Mean Absolute Error (MAE) and R².
7. Save the trained model with Joblib.
8. Load it in the Streamlit website and predict from user inputs.

## Limitations
- This is a demonstration only because the bundled data is synthetic.
- It does not include vehicle condition, location, exact model/variant, service history, accident history, or live market prices.
- Predictions should not be used to make buying/selling decisions.
