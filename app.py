"""
Flask Stock Prediction API
===========================

A RESTful API for stock price prediction using machine learning algorithms.
Supports multiple ML models (Random Forest, Linear Regression, SVR) and 
provides historical data analysis with next-day price predictions.

Author: [Your Name]
Version: 1.0.0
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import os
import json

app = Flask(__name__)
CORS(app)

# Supported stocks and models
STOCKS = {
    'AAL': 'American Airlines Group',
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.'
}

MODELS = {
    'random_forest': 'Random Forest',
    'linear_regression': 'Linear Regression',
    'svr': 'Support Vector Regression'
}


def download_stock_data(symbol, period='2y'):
    """
    Download historical stock data from Yahoo Finance.
    
    Fetches OHLCV (Open, High, Low, Close, Volume) data along with
    dividends and stock splits for the specified symbol and time period.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL').
    period : str, optional
        Time period for historical data. Valid options include:
        '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'.
        Default is '2y'.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with DatetimeIndex containing columns:
        - Open: Opening price
        - High: Highest price
        - Low: Lowest price
        - Close: Closing price
        - Volume: Trading volume
        - Dividends: Dividend payments
        - Stock Splits: Stock split ratios
    
    Raises
    ------
    Exception
        If the symbol is invalid or data cannot be retrieved from Yahoo Finance.
    
    Examples
    --------
    >>> df = download_stock_data('AAPL', period='1y')
    >>> print(df.head())
    >>> print(f"Data shape: {df.shape}")
    
    Notes
    -----
    - Requires active internet connection
    - Data availability depends on Yahoo Finance API
    - Historical data may have gaps for non-trading days
    """
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df


def prepare_features(df):
    """
    Engineer features from raw stock data for machine learning.
    
    Creates temporal features, technical indicators, lagged values, and
    moving averages from OHLCV data. Removes rows with missing values
    after feature creation.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Raw stock data with DatetimeIndex and columns:
        Open, High, Low, Close, Volume.
    
    Returns
    -------
    pandas.DataFrame
        Enhanced DataFrame with original columns plus:
        - Day: Day of week (0=Monday, 6=Sunday)
        - Month: Month number (1-12)
        - Year: Year
        - Price_Up: Binary indicator (1 if Close > Open, else 0)
        - High_Low_Pct: Percentage range (High-Low)/Low * 100
        - Open_Close_Pct: Percentage change (Close-Open)/Open * 100
        - Close_Lag_[1,2,3,5]: Lagged closing prices
        - Volume_Lag_[1,2,3,5]: Lagged volumes
        - MA_[5,10,20]: Moving averages for 5, 10, and 20 days
    
    Examples
    --------
    >>> raw_df = download_stock_data('AAPL', '6mo')
    >>> features_df = prepare_features(raw_df)
    >>> print(features_df.columns.tolist())
    >>> print(f"Original rows: {len(raw_df)}, After features: {len(features_df)}")
    
    Notes
    -----
    - Drops approximately 20 rows at the start due to lag features and moving averages
    - All NaN values are removed to ensure clean training data
    - DataFrame is copied to avoid modifying the original
    """
    df = df.copy()
    df['Day'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['Price_Up'] = (df['Close'] > df['Open']).astype(int)
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Low'] * 100
    df['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
    
    for window in [5, 10, 20]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    
    df = df.dropna()
    return df


def train_model(df, model_type='random_forest'):
    """
    Train a machine learning model to predict next-day opening prices.
    
    Splits data into 80/20 train/test sets, scales features, trains the
    specified model, and evaluates performance on test data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Feature-engineered DataFrame from prepare_features().
    model_type : str, optional
        Machine learning algorithm to use. Options:
        - 'random_forest': Random Forest Regressor (default)
        - 'linear_regression': Linear Regression
        - 'svr': Support Vector Regression with RBF kernel
    
    Returns
    -------
    tuple
        (model, scaler, X_test, y_test, y_pred, mse, train_size) where:
        - model: Trained scikit-learn model object
        - scaler: Fitted StandardScaler for feature normalization
        - X_test: Test feature set (pandas.DataFrame)
        - y_test: True test target values (pandas.Series)
        - y_pred: Model predictions on test set (numpy.ndarray)
        - mse: Mean Squared Error on test set (float)
        - train_size: Number of training samples (int)
    
    Raises
    ------
    ValueError
        If model_type is not one of: 'random_forest', 'linear_regression', 'svr'.
    
    Examples
    --------
    >>> df = prepare_features(download_stock_data('AAPL', '1y'))
    >>> model, scaler, X_test, y_test, y_pred, mse, train_size = train_model(df, 'random_forest')
    >>> print(f"MSE: {mse:.2f}, Training samples: {train_size}")
    >>> print(f"Test accuracy: R² = {1 - mse/np.var(y_test):.3f}")
    
    Notes
    -----
    - Target variable is next day's opening price (Open shifted -1)
    - Features exclude: Open, High, Low, Close, Volume, Dividends, Stock Splits
    - Uses 80/20 chronological split (no shuffling to preserve temporal order)
    - StandardScaler normalizes features to zero mean and unit variance
    - Random Forest uses 100 estimators with fixed random_state=42
    - SVR uses RBF kernel with C=1.0, epsilon=0.1
    """
    features = df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], axis=1, errors='ignore')
    target = df['Open'].shift(-1)
    
    features = features.iloc[:-1]
    target = target.iloc[:-1]
    
    split_idx = int(len(features) * 0.8)
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'svr':
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, scaler, X_test, y_test, y_pred, mse, len(X_train)


def predict_next_day(model, scaler, latest_data):
    """
    Predict the next day's opening price using the trained model.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model from train_model().
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler from train_model().
    latest_data : pandas.DataFrame
        Single-row DataFrame containing the most recent data with
        all engineered features.
    
    Returns
    -------
    float
        Predicted opening price for the next trading day.
    
    Examples
    --------
    >>> df = prepare_features(download_stock_data('AAPL', '6mo'))
    >>> model, scaler, *_ = train_model(df)
    >>> latest = df.iloc[-1:].copy()
    >>> next_price = predict_next_day(model, scaler, latest)
    >>> print(f"Predicted next opening price: ${next_price:.2f}")
    
    Notes
    -----
    - latest_data must have same features used during training
    - Features are automatically scaled using the provided scaler
    - Returns single prediction even if latest_data has multiple rows
    """
    latest_features = latest_data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], axis=1, errors='ignore')
    latest_features_scaled = scaler.transform(latest_features)
    prediction = model.predict(latest_features_scaled)
    return prediction[0]


@app.route('/')
def index():
    """
    Render the main application page.
    
    Returns
    -------
    str
        Rendered HTML template with available stocks and models.
    
    Notes
    -----
    Requires 'templates/index.html' to exist in the project structure.
    """
    return render_template('index.html', stocks=STOCKS, models=MODELS)


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for stock price prediction.
    
    Accepts stock symbol, ML algorithm, and time period via POST request.
    Downloads data, trains model, generates predictions, and returns
    comprehensive results including historical comparison and next-day forecast.
    
    Request Parameters
    ------------------
    Content-Type: application/json or application/x-www-form-urlencoded
    
    Body Parameters:
        stock : str, optional
            Stock ticker symbol (default: 'AAL'). Must be in STOCKS dict.
        algorithm : str, optional
            ML model type (default: 'random_forest'). Options:
            'random_forest', 'linear_regression', 'svr'.
        period : str, optional
            Historical data period (default: '6mo'). Valid values:
            '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'.
    
    Returns
    -------
    JSON Response (200 OK):
        {
            "stock": str,              # Stock ticker symbol
            "stock_name": str,         # Full company name
            "model": str,              # Model name used
            "mse": float,              # Mean Squared Error on test set
            "last_actual_open": float, # Most recent opening price
            "next_day_prediction": float, # Predicted next opening price
            "dates": list[str],        # Last 20 test dates (YYYY-MM-DD)
            "actual_prices": list[float], # Last 20 actual prices
            "predicted_prices": list[float], # Last 20 predicted prices
            "data_points": int,        # Total data points used
            "training_points": int,    # Number of training samples
            "recent_dates": list[str]  # Last 10 dates in dataset
        }
    
    JSON Response (500 Error):
        {
            "error": str  # Error message describing what went wrong
        }
    
    Examples
    --------
    cURL request:
    ```bash
    curl -X POST http://localhost:5000/predict \
         -H "Content-Type: application/json" \
         -d '{"stock": "AAPL", "algorithm": "random_forest", "period": "1y"}'
    ```
    
    Python request:
    ```python
    import requests
    
    response = requests.post('http://localhost:5000/predict',
                            json={'stock': 'TSLA',
                                  'algorithm': 'linear_regression',
                                  'period': '6mo'})
    data = response.json()
    print(f"Next day prediction: ${data['next_day_prediction']:.2f}")
    print(f"MSE: {data['mse']:.2f}")
    ```
    
    JavaScript fetch:
    ```javascript
    fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            stock: 'GOOGL',
            algorithm: 'svr',
            period: '1y'
        })
    })
    .then(response => response.json())
    .then(data => console.log(data));
    ```
    
    Raises
    ------
    500 Internal Server Error
        - Invalid stock symbol or period
        - Network error accessing Yahoo Finance
        - Insufficient data for the requested period
        - Model training failure
        - Invalid algorithm type
    
    Notes
    -----
    - Requires active internet connection for Yahoo Finance data
    - Processing time varies by period (typically 2-10 seconds)
    - Returns last 20 predictions for visualization purposes
    - Model is trained fresh on each request (not cached)
    - All prices in USD
    - Predictions are for next trading day, not calendar day
    
    Security Considerations
    -----------------------
    - Input validation: Only predefined stocks and models accepted
    - CORS enabled for cross-origin requests
    - No authentication required (consider adding for production)
    """
    if request.content_type == 'application/json':
        data = request.json
    else:
        data = request.form
    
    stock_symbol = data.get('stock', 'AAL')
    model_type = data.get('algorithm', 'random_forest')
    period = data.get('period', '6mo')
    
    try:
        df = download_stock_data(stock_symbol, period)
        df = prepare_features(df)
        
        model, scaler, X_test, y_test, y_pred, mse, training_points = train_model(df, model_type)
        
        latest_data = df.iloc[-1:].copy()
        next_day_prediction = predict_next_day(model, scaler, latest_data)
        
        plot_dates = X_test.index.strftime('%Y-%m-%d').tolist()[-20:]
        actual_prices = y_test.values[-20:]
        predicted_prices = y_pred[-20:]
        
        recent_dates = df.index.strftime('%Y-%m-%d').tolist()[-10:]
        
        response = {
            'stock': stock_symbol,
            'stock_name': STOCKS.get(stock_symbol, stock_symbol),
            'model': MODELS.get(model_type, model_type),
            'mse': mse,
            'last_actual_open': df['Open'].iloc[-1],
            'next_day_prediction': next_day_prediction,
            'dates': plot_dates,
            'actual_prices': actual_prices.tolist(),
            'predicted_prices': predicted_prices.tolist(),
            'data_points': len(df),
            'training_points': training_points,
            'recent_dates': recent_dates
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    """
    Run the Flask development server.
    
    Configuration
    -------------
    - Debug mode: Enabled (auto-reload on code changes)
    - Port: 5000
    - Host: localhost (127.0.0.1)
    
    Usage
    -----
    Run from command line:
        python app.py
    
    Access at:
        http://localhost:5000
    
    Notes
    -----
    - Debug mode should be disabled in production
    - Consider using a production WSGI server (gunicorn, uWSGI)
    - Set environment variables for configuration in production
    """
    app.run(debug=True, port=5000)