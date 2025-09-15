from flask import Flask, render_template, jsonify
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "dbname": "nepal_stock",
    "user": "postgres",
    "password": "amir",
    "port": 5432
}

def connect_db():
    return psycopg2.connect(**DB_CONFIG)

def get_all_companies():
    """Get all companies from database"""
    conn = connect_db()
    query = "SELECT id, company_name, company_tag FROM company ORDER BY company_tag"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def get_price_history(company_id=None, limit=None):
    """Get price history data"""
    conn = connect_db()
    
    if company_id:
        query = """
        SELECT ph.*, c.company_name, c.company_tag 
        FROM price_history ph 
        JOIN company c ON ph.company_id = c.id 
        WHERE ph.company_id = %s 
        ORDER BY ph.date DESC
        """
        params = (company_id,)
        if limit:
            query += f" LIMIT {limit}"
        df = pd.read_sql(query, conn, params=params)
    else:
        query = """
        SELECT ph.*, c.company_name, c.company_tag 
        FROM price_history ph 
        JOIN company c ON ph.company_id = c.id 
        ORDER BY c.company_tag, ph.date DESC
        """
        df = pd.read_sql(query, conn)
    
    conn.close()
    return df

def create_features(df):
    """Create features for ML model"""
    df = df.copy()
    df = df.sort_values('date')
    
    # Fill missing values
    numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    # Technical indicators
    df['sma_5'] = df['close_price'].rolling(window=min(5, len(df))).mean()
    df['sma_10'] = df['close_price'].rolling(window=min(10, len(df))).mean()
    df['price_change'] = df['close_price'].pct_change()
    df['volatility'] = df['close_price'].rolling(window=min(5, len(df))).std()
    df['high_low_pct'] = (df['high_price'] - df['low_price']) / df['close_price']
    df['volume_change'] = df['volume'].pct_change() if 'volume' in df.columns else 0
    
    # Lag features
    for lag in [1, 2, 3, 5]:
        if lag < len(df):
            df[f'close_lag_{lag}'] = df['close_price'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
    
    # Moving averages ratio
    df['sma_ratio'] = df['sma_5'] / df['sma_10']
    
    # Fill remaining NaN values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df

def predict_stock_prices(company_id, days=5):
    """Predict stock prices for next 5 days"""
    try:
        # Get historical data (last 30 records for better training)
        df = get_price_history(company_id, limit=30)
        
        if len(df) < 10:
            return None, f"Insufficient data (only {len(df)} records)"
        
        df = df.sort_values('date')
        df = create_features(df)
        
        # Feature columns - only use available features
        base_features = ['open_price', 'high_price', 'low_price']
        if 'volume' in df.columns:
            base_features.append('volume')
        
        feature_cols = base_features + [
            'sma_5', 'sma_10', 'price_change', 'volatility', 'high_low_pct'
        ]
        
        # Add lag features that exist
        for lag in [1, 2, 3, 5]:
            if f'close_lag_{lag}' in df.columns:
                feature_cols.append(f'close_lag_{lag}')
            if f'volume_lag_{lag}' in df.columns and 'volume' in df.columns:
                feature_cols.append(f'volume_lag_{lag}')
        
        if 'sma_ratio' in df.columns:
            feature_cols.append('sma_ratio')
        
        # Remove rows with NaN values
        df_clean = df[feature_cols + ['close_price']].dropna()
        
        if len(df_clean) < 5:
            return None, f"Insufficient clean data after feature engineering ({len(df_clean)} rows)"
        
        X = df_clean[feature_cols]
        y = df_clean['close_price']
        
        # Use all data for training if we have limited data
        if len(X) <= 8:
            X_train, X_val = X, X[-2:]  # Use last 2 for validation
            y_train, y_val = y, y[-2:]
        else:
            # Use last 80% for training
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train model with simpler parameters for small datasets
        n_estimators = min(50, len(X_train) * 5)
        max_depth = min(5, len(X_train) // 2)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=42, 
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1
        )
        model.fit(X_train_scaled, y_train)
        
        # Validate model
        val_pred = model.predict(X_val_scaled)
        mae = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        # Predict next 5 days
        predictions = []
        last_row = df_clean.iloc[-1].copy()
        
        for day in range(days):
            # Prepare features for prediction
            X_pred = last_row[feature_cols].values.reshape(1, -1)
            X_pred_scaled = scaler.transform(X_pred)
            
            # Predict
            pred_price = model.predict(X_pred_scaled)[0]
            predictions.append(max(pred_price, 0.01))  # Ensure positive price
            
            # Update last_row for next prediction
            if 'close_lag_3' in last_row:
                last_row['close_lag_3'] = last_row.get('close_lag_2', last_row['close_price'])
            if 'close_lag_2' in last_row:
                last_row['close_lag_2'] = last_row.get('close_lag_1', last_row['close_price'])
            if 'close_lag_1' in last_row:
                last_row['close_lag_1'] = last_row['close_price']
            
            last_row['close_price'] = pred_price
            
            # Update other features
            if 'sma_5' in last_row:
                last_row['sma_5'] = (last_row['sma_5'] * 4 + pred_price) / 5
            if 'price_change' in last_row and 'close_lag_1' in last_row:
                last_row['price_change'] = (pred_price - last_row['close_lag_1']) / last_row['close_lag_1']
        
        return {
            'predictions': predictions,
            'mae': mae,
            'rmse': rmse,
            'last_price': float(df['close_price'].iloc[-1]),
            'company_tag': df['company_tag'].iloc[0],
            'company_name': df['company_name'].iloc[0]
        }, None
        
    except Exception as e:
        return None, f"Error in prediction: {str(e)}"

def calculate_investment_recommendation():
    companies = get_all_companies()
    all_predictions = {}
    
    print("Calculating recommendations for all companies...")
    
    for _, company in companies.iterrows():
        pred_result, error = predict_stock_prices(company['id'])
        if pred_result:
            all_predictions[company['company_tag']] = pred_result
            print(f"✓ Predictions ready for {company['company_tag']}")
        else:
            print(f"✗ Failed for {company['company_tag']}: {error}")
    
    if not all_predictions:
        return {}
    
    # Calculate expected returns for each stock each day
    recommendations = {}
    used_stocks = set()
    
    for day in range(5):
        day_recommendations = []
        
        for tag, pred_data in all_predictions.items():
            if tag in used_stocks:  # Skip if already recommended
                continue
                
            if day < len(pred_data['predictions']):
                current_price = pred_data['last_price']
                predicted_price = pred_data['predictions'][day]
                expected_return = (predicted_price - current_price) / current_price * 100
                
                # Calculate confidence score
                confidence = max(0, 100 - (pred_data['mae'] / current_price * 100))
                
                day_recommendations.append({
                    'stock': tag,
                    'company_name': pred_data['company_name'],
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'expected_return': expected_return,
                    'mae': pred_data['mae'],
                    'confidence': confidence,
                    'score': expected_return * (confidence / 100)  # Combined score
                })
        
        # Sort by combined score (return * confidence)
        day_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Get the best available stock for this day
        if day_recommendations:
            best_stock = day_recommendations[0]
            used_stocks.add(best_stock['stock'])  # Mark as used
            
            date = (datetime.now() + timedelta(days=day+1)).strftime('%Y-%m-%d')
            recommendations[date] = best_stock
    
    return recommendations

@app.route('/')
def index():
    """Main dashboard"""
    companies = get_all_companies()
    all_history = get_price_history()
    total_records = all_history.shape[0]
    
    # Get latest prices for each company
    latest_prices = {}
    for _, company in companies.iterrows():
        company_history = get_price_history(company['id'], limit=1)
        if len(company_history) > 0:
            latest_prices[company['company_tag']] = {
                'price': company_history.iloc[0]['close_price'],
                'date': company_history.iloc[0]['date']
            }
    
    return render_template('index.html', 
                         companies=companies.to_dict('records'),
                         total_records=total_records,
                         latest_prices=latest_prices)

@app.route('/api/companies')
def api_companies():
    """API endpoint for companies"""
    companies = get_all_companies()
    return jsonify(companies.to_dict('records'))

@app.route('/api/price_history/<int:company_id>')
def api_price_history(company_id):
    """API endpoint for price history"""
    df = get_price_history(company_id)
    df['date'] = df['date'].astype(str)
    # Convert numeric columns to ensure JSON serialization
    numeric_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return jsonify(df.to_dict('records'))

@app.route('/api/predictions/<int:company_id>')
def api_predictions(company_id):
    """API endpoint for predictions"""
    result, error = predict_stock_prices(company_id)
    if error:
        return jsonify({'error': error}), 400
    
    # Add dates to predictions
    predictions_with_dates = []
    for i, price in enumerate(result['predictions']):
        date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
        predictions_with_dates.append({
            'date': date,
            'predicted_price': round(price, 2),
            'day': i + 1
        })
    
    return jsonify({
        'company_info': {
            'tag': result['company_tag'],
            'name': result['company_name'],
            'last_price': round(result['last_price'], 2)
        },
        'predictions': predictions_with_dates,
        'model_metrics': {
            'mae': round(result['mae'], 2),
            'rmse': round(result['rmse'], 2)
        }
    })

@app.route('/api/recommendations')
def api_recommendations():
    try:
        recommendations = calculate_investment_recommendation()
        # Round numeric values for better display
        for date, rec in recommendations.items():
            rec['current_price'] = round(rec['current_price'], 2)
            rec['predicted_price'] = round(rec['predicted_price'], 2)
            rec['expected_return'] = round(rec['expected_return'], 2)
            rec['confidence'] = round(rec['confidence'], 2)
            rec['mae'] = round(rec['mae'], 2)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predictions')
def predictions_page():
    companies = get_all_companies()
    return render_template('predictions.html', companies=companies.to_dict('records'))

@app.route('/recommendations')
def recommendations_page():
    return render_template('recommendations.html')

@app.route('/database')
def database_page():
    companies = get_all_companies()
    return render_template('database.html', companies=companies.to_dict('records'))

if __name__ == '__main__':
    print("Nepal Stock Predictor")
    print("Starting server...")
    print("Open: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)