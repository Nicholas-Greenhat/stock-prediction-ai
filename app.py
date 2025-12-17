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
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df

def prepare_features(df):
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
    latest_features = latest_data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'], axis=1, errors='ignore')
    latest_features_scaled = scaler.transform(latest_features)
    prediction = model.predict(latest_features_scaled)
    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html', stocks=STOCKS, models=MODELS)

@app.route('/predict', methods=['POST'])
def predict():
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
    app.run(debug=True, port=5000)
