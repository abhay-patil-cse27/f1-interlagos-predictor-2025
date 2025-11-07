# 🏎️ F1 Brazilian Grand Prix (São Paulo – Interlagos) Prediction

This repository contains an end-to-end machine learning project that predicts the **finishing positions for the 2025 Formula 1 São Paulo Grand Prix (Interlagos)** using **manually collected and refined race data** from the past decade (2014–2024).

---

## 📘 Overview

Due to the discontinuation of major public Formula 1 APIs such as **FastF1** and **OpenF1** after 2024 (callback and rate-limit restrictions), all race data used in this project was **manually scraped and preprocessed** into clean CSV files.

This project leverages advanced regression models to forecast the finishing positions of all drivers in the **2025 Brazilian GP**. The model considers factors such as driver performance trends, constructor efficiency, qualifying outcomes, pit stop timing, and circuit characteristics.

> 🏁 **Predicted winner for the 2025 São Paulo GP (Interlagos):**  
> **Max Verstappen** — *Red Bull Racing*  

---

## 🧠 Model Summary

- **Algorithm used:** Gradient Boosting Regressor with Ridge regularization  
- **Objective:** Predict finishing position for each driver  
- **Training data:** Combined and feature-engineered data from multiple seasons (2014–2024)  
- **Evaluation metrics:** MSE, RMSE, MAE, R²  
- **Frameworks used:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`  

The notebook trains, evaluates, and visualizes model performance, and the final model is saved as:

```
models/gradient_boosting_regressor_f1_prediction_model.joblib
```

---

## 🧩 Data Files

All input CSV files are manually scraped and cleaned before use.  
They are stored in the `data/` folder:

```
data/
├─ circuits.csv
├─ constructors.csv
├─ constructor_results.csv
├─ constructor_standings.csv
├─ drivers.csv
├─ driver_standings.csv
├─ lap_times.csv
├─ pit_stops.csv
├─ qualifying.csv
├─ races.csv
├─ results.csv
├─ seasons.csv
├─ sprint_results.csv
└─ status.csv
```

These datasets were merged and processed into a comprehensive training frame that represents driver, constructor, and race-level statistics.

---

## 📊 Notebook Workflow

The Jupyter notebook `Brazilian_GP.ipynb` includes the following steps:

1. **Data ingestion** – Load all CSV files using `pandas`  
2. **Merging & cleaning** – Join datasets by race, driver, and constructor IDs  
3. **Feature engineering** –  
   - Rolling averages of driver performance  
   - Circuit-specific success rates  
   - Constructor standings and pit efficiency  
   - Recent finishing trends and qualifying form  
4. **Model training** – Gradient Boosting + Ridge ensemble  
5. **Evaluation** – Compute MSE, RMSE, MAE, and R²  
6. **Prediction** – Output expected 2025 Brazilian GP driver standings  

---

## ⚙️ Requirements

Create a `requirements.txt` file with the following dependencies:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
jupyterlab
```

Install all packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/abhay-patil-cse27/f1-brazilian-gp-prediction.git
   cd f1-brazilian-gp-prediction
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # for Windows
   source venv/bin/activate     # for macOS/Linux
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   Open and run `Brazilian_GP.ipynb`.

4. To use the trained model directly:
   ```python
   import joblib, pandas as pd

   model = joblib.load("models/gradient_boosting_regressor_f1_prediction_model.joblib")
   X_test = pd.read_csv("data/sample_input.csv")   # must match training features
   predictions = model.predict(X_test)

   print(predictions)
   ```

---

## 📈 Model Performance

| Metric | Description | Result (see notebook output) |
|---------|-------------|------------------------------|
| **MSE** | Mean Squared Error | 2.9807 |
| **RMSE** | Root Mean Squared Error | 16.4161 |
| **MAE** | Mean Absolute Error | 4.0517 |
| **R²** | Coefficient of Determination | 0.5329 |

*(Exact numeric values are shown in the notebook output cells.)*

---

## 🗂️ Repository Structure

```
f1-brazilian-gp-prediction/
│
├─ data/                           # Input CSVs
├─ models/
│   └─ gradient_boosting_regressor_f1_prediction_model.joblib
├─ notebooks/
│   └─ Brazilian_GP.ipynb
├─ scripts/
│   └─ predict.py (optional inference script)
├─ .gitignore
├─ requirements.txt
└─ README.md
```

---

## ⚠️ Notes

- Both **FastF1** and **OpenF1** APIs stopped functioning after 2024 due to data access restrictions.  
  Hence, all required CSVs were built manually via data scraping and formatting.
- The model focuses on **driver finishing position prediction** for educational and research use.  
- Future improvements may include extending to other circuits and incorporating real-time telemetry (if API access resumes).

---

## 📄 License

Not Needed.

---

## 👨‍💻 Author

**Abhay Patil**  
B.Tech Computer Science & Engineering  
KIT’s College of Engineering (Empowered Autonomous), Kolhapur  
GitHub: [@abhay-patil-cse27](https://github.com/abhay-patil-cse27)
