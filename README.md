# Stock Market Prediction AI

A web application that predicts stock prices using machine learning algorithms.

## Features

- Fetches real stock data from Yahoo Finance API
- Implements multiple ML algorithms (Random Forest, Linear Regression, SVR)
- Provides next-day opening price predictions
- Interactive visualization of actual vs predicted prices
- Clean, responsive web interface

## Installation

1. Create a virtual environment: python -m venv venv
2. Activate the virtual environment: env\Scripts\activate
3. Install dependencies: pip install -r requirements.txt
4. Run the application: python app.py
5. Open your browser and go to http://localhost:5000

## Usage

1. Select a stock from the dropdown menu
2. Choose a machine learning algorithm
3. Select the data period to use for training
4. Click \"Fetch Data & Predict\"
5. View the results including MSE, predictions, and visualizations

## Project Structure

- pp.py - Main Flask application
- equirements.txt - Python dependencies
- 	emplates/ - HTML templates
- static/ - CSS and JavaScript files
- data/ - Raw and processed data
- 
otebooks/ - Jupyter notebooks for experimentation
- models/ - Saved trained models

## Algorithms

- Random Forest Regressor
- Linear Regression
- Support Vector Regression (SVR)
