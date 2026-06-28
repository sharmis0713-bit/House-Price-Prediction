# 🏠 House Price Prediction
### SyntecHub Internship — Project 1

A Machine Learning project that predicts house prices using **Linear Regression** on the California Housing Dataset.

---

## 📌 Project Overview

This project demonstrates a complete end-to-end Machine Learning pipeline — from loading and exploring data, to training a model, evaluating its performance, and saving it for future predictions.

---

## 📂 Project Structure

```
House-Price-Prediction/
│
├── house_price_prediction.py   # Main Python script
├── model/
│   └── house_price_model.pkl   # Saved trained model
├── price_distribution.png      # EDA plot 1
├── correlation_heatmap.png     # EDA plot 2
├── actual_vs_predicted.png     # Model evaluation plot
├── requirements.txt            # Required libraries
└── README.md                   # Project documentation
```

---

## 📊 Dataset

- **Source:** California Housing Dataset (built into scikit-learn)
- **Origin:** 1990 U.S. Census data for California
- **Size:** 20,640 samples × 8 features

| Feature | Description |
|---|---|
| MedInc | Median income in the block |
| HouseAge | Median house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Block population |
| AveOccup | Average house occupancy |
| Latitude | Block latitude |
| Longitude | Block longitude |
| **MedHouseVal** | **Target — Median house value ($100,000s)** |

---

## ⚙️ Tech Stack

- Python 3.14
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/sharmis0713-bit/House-Price-Prediction.git
cd House-Price-Prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the script**
```bash
python house_price_prediction.py
```

---

## 📈 Results

| Metric | Value |
|---|---|
| RMSE | 0.7456 |
| R² Score | 0.5758 |

- **RMSE of 0.75** → predictions are off by ~$74,558 on average
- **R² of 0.58** → the model explains **57.6% of house price variation**

---

## 🔍 Key Findings

- **Median Income** is the strongest predictor of house price (correlation: 0.69)
- **AveBedrms** has the highest positive coefficient (0.78) — more bedrooms = higher price
- **Location** (Latitude/Longitude) negatively impacts price in this California-specific dataset
- Linear Regression performs reasonably well but a non-linear model (e.g. Random Forest) would improve R² significantly

---

## 📉 Visualisations

| Plot | Description |
|---|---|
| `price_distribution.png` | Distribution of median house values |
| `correlation_heatmap.png` | Feature correlation matrix |
| `actual_vs_predicted.png` | Model predictions vs actual prices |

---

## 🙋 Author

**Sharmi S**
B.Sc. Artificial Intelligence & Machine Learning
Hindusthan College of Arts and Science, Coimbatore

---

*Project completed as part of SyntecHub ML Internship*
