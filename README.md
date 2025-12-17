# 📈 Stock Market Prediction AI

**A Machine Learning–Powered Web Application for Stock Price Forecasting**

---

## 1. Project Overview

### **Description**

**Stock Market Prediction AI** is a full-stack machine learning web application that predicts **next-day stock opening prices** using historical market data and multiple regression models.

The application combines:

* Real-time financial data from **Yahoo Finance**
* Feature-engineered time-series data
* Multiple machine learning algorithms
* A Flask REST API backend
* A responsive, interactive frontend built with **Bootstrap**, **Chart.js**, and vanilla JavaScript

This project demonstrates how to **bridge data science and web engineering** by deploying ML models into an interactive web interface.

---

## 2. Key Features

* 📊 Fetches real historical stock data using Yahoo Finance
* 🤖 Supports multiple ML algorithms:

  * Random Forest Regressor
  * Linear Regression
  * Support Vector Regression (SVR)
* 📈 Predicts **next trading day opening price**
* 📉 Displays **Mean Squared Error (MSE)** for model evaluation
* 📊 Interactive chart showing **Actual vs Predicted prices**
* 🧠 Automatic feature engineering (lags, moving averages, temporal features)
* 🖥️ Clean, responsive UI with loading indicators
* 🔁 Graceful fallback to simulated data if API fails

---

## 3. Technology Stack

### **Backend**

* Python 3.8+
* Flask
* Flask-CORS
* Yahoo Finance (`yfinance`)
* Scikit-learn
* Pandas & NumPy

### **Frontend**

* HTML5
* CSS3
* Bootstrap 5
* JavaScript (ES6)
* Chart.js

### **Machine Learning**

* Random Forest Regressor
* Linear Regression
* Support Vector Regression (RBF kernel)
* StandardScaler for feature normalization

---

## 4. System Requirements

### **Operating System**

* Windows (tested)
* macOS
* Linux

### **Required Tools**

* Python 3.8 or higher
* pip
* Internet connection (for Yahoo Finance data)

---

## 5. Installation & Setup

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/your-username/stock-market-prediction-ai.git
cd stock-market-prediction-ai
```

### **Step 2: Create a Virtual Environment**

```bash
python -m venv venv
```

### **Step 3: Activate the Virtual Environment**

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### **Step 4: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 5: Run the Application**

```bash
python app.py
```

### **Step 6: Open in Browser**

```
http://localhost:5000
```

---

## 6. Application Usage

### **Using the Web Interface**

1. Select a **stock ticker** (AAPL, MSFT, GOOGL, etc.)
2. Choose a **machine learning algorithm**
3. Select the **historical data period**
4. Click **“Fetch Data & Predict”**
5. View:

   * Model evaluation (MSE)
   * Last actual opening price
   * Predicted next-day opening price
   * Interactive price comparison chart
   * Accuracy and error metrics

---

## 7. API Documentation

### **Endpoint**

```
POST /predict
```

### **Request Body**

```json
{
  "stock": "AAPL",
  "algorithm": "random_forest",
  "period": "1y"
}
```

### **Response Example**

```json
{
  "stock": "AAPL",
  "stock_name": "Apple Inc.",
  "model": "Random Forest",
  "mse": 0.423156,
  "last_actual_open": 189.32,
  "next_day_prediction": 191.04,
  "dates": ["2024-10-01", "2024-10-02"],
  "actual_prices": [188.4, 189.3],
  "predicted_prices": [187.9, 190.1],
  "data_points": 245,
  "training_points": 196,
  "recent_dates": ["2024-10-10", "2024-10-11"]
}
```

---

## 8. Machine Learning Pipeline

### **Data Processing**

* Historical OHLCV data fetched via Yahoo Finance
* Feature engineering includes:

  * Day, Month, Year
  * Price movement indicators
  * Lagged closing prices and volume
  * Moving averages (5, 10, 20 days)

### **Training Strategy**

* Chronological 80/20 train-test split
* Feature scaling using StandardScaler
* Model trained on each request (no caching)

### **Evaluation Metric**

* Mean Squared Error (MSE)

---

## 9. Project Structure

```
stock-market-prediction-ai/
│
├── app.py                 # Flask API and ML logic
├── requirements.txt       # Version-pinned dependencies
│
├── templates/
│   └── index.html         # Frontend UI
│
├── static/
│   ├── css/               # Styling (if extended)
│   └── js/                # Client-side logic
│
├── data/                  # Optional data storage
├── notebooks/             # Jupyter experiments
├── models/                # Saved models (optional)
│
└── README.md
```

---

## 10. Configuration Options

* Add new stocks in the `STOCKS` dictionary
* Tune ML hyperparameters in `train_model()`
* Increase historical window size
* Cache trained models for performance
* Disable debug mode for production

---

## 11. Common Issues & Troubleshooting

| Issue            | Cause                | Solution             |
| ---------------- | -------------------- | -------------------- |
| App won’t start  | Virtual env inactive | Activate environment |
| `/predict` fails | No internet          | Check connection     |
| Poor predictions | Short data period    | Increase data window |
| 404 error        | Flask not running    | Restart server       |

---

## 12. Contributing Guidelines

Contributions are welcome 🚀

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/your-feature
```

3. Commit changes
4. Push to your fork
5. Open a Pull Request

---

## 13. License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute this project with attribution.

---

## 14. Future Enhancements

* LSTM / Deep Learning models
* Real-time streaming prices
* Model caching
* Cloud deployment (AWS / Render / Fly.io)
* User authentication
* Technical indicators (RSI, MACD, Bollinger Bands)

---

### ✅ Status

**Complete & Functional MVP**
Ready for:

* Portfolio showcase
* Hackathons
* Academic projects
* ML demos

---

If you want next:

* 🚀 **Production deployment guide**
* 🧠 **Deep learning extension**
* 🧪 **Unit tests & CI**
* 📊 **Model comparison dashboard**

