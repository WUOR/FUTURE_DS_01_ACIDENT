import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

def load_data():
    """Load and preprocess accident data"""
    df = pd.read_csv('../data/accident_data.csv')
    
    # Feature engineering
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5,6]).astype(int)
    
    # Convert categorical features
    df = pd.get_dummies(df, columns=['WeatherConditions', 'Cause'])
    
    return df

def train_model(df):
    """Train predictive model for accident severity"""
    X = df.drop(['AccidentID', 'Date', 'Location', 'Severity'], axis=1)
    y = df['Severity']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Model MAE: {mae:.2f}")
    
    return model

def predict_risk(model, location_data):
    """Predict risk score for new locations"""
    # location_data should be a DataFrame with same features as training data
    return model.predict(location_data)

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_data()
    
    print("Training predictive model...")
    model = train_model(df)
    
    # Save model for Power BI integration
    joblib.dump(model, 'accident_severity_model.pkl')
    print("Model saved to accident_severity_model.pkl")