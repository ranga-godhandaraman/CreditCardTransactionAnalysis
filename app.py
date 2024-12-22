from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv(r'C:\Users\ADMIN\Documents\project-1\creditcard.csv')
le = LabelEncoder()

# Prepare the data
df_ml = df.copy()
df_ml['Gender'] = le.fit_transform(df['Gender'])
df_ml['Card_Type'] = le.fit_transform(df['Card Type'])
df_ml['Exp_Type'] = le.fit_transform(df['Exp Type'])
df_ml['City'] = le.fit_transform(df['City'])
df_ml['Month'] = pd.to_datetime(df['Date']).dt.month
df_ml['Year'] = pd.to_datetime(df['Date']).dt.year

# Train models (simplified versions for demo)
def train_models():
    # Card Recommendation Model
    features = ['Gender', 'Amount', 'Exp_Type', 'Month', 'City']
    X = df_ml[features]
    y = df_ml['Card_Type']
    card_model = xgb.XGBClassifier(random_state=42)
    card_model.fit(X, y)
    
    # Risk Assessment Model
    df_ml['Avg_Transaction'] = df_ml.groupby('Card_Type')['Amount'].transform('mean')
    df_ml['Transaction_Ratio'] = df_ml['Amount'] / df_ml['Avg_Transaction']
    df_ml['Risk_Level'] = pd.qcut(df_ml['Transaction_Ratio'], q=3, labels=['Low', 'Medium', 'High'])
    risk_model = RandomForestClassifier(random_state=42)
    risk_features = ['Amount', 'Gender', 'Card_Type', 'Exp_Type', 'Month']
    risk_model.fit(df_ml[risk_features], le.fit_transform(df_ml['Risk_Level']))
    
    return card_model, risk_model

# Train models
card_model, risk_model = train_models()

@app.route('/')
def home():
    cities = sorted(df['City'].unique())
    card_types = sorted(df['Card Type'].unique())
    exp_types = sorted(df['Exp Type'].unique())
    return render_template('index.html', 
                         cities=cities,
                         card_types=card_types,
                         exp_types=exp_types)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        
        # Prepare input data
        input_data = {
            'Gender': 1 if data['gender'] == 'M' else 0,
            'Amount': float(data['amount']),
            'Exp_Type': le.transform([data['exp_type']])[0],
            'Month': int(data['month']),
            'City': le.transform([data['city']])[0]
        }
        
        # Make predictions
        input_df = pd.DataFrame([input_data])
        card_prediction = le.inverse_transform([card_model.predict(input_df)[0]])[0]
        risk_prediction = le.inverse_transform([risk_model.predict(input_df)[0]])[0]
        
        # Get city statistics
        city_stats = df[df['City'] == data['city']]['Amount'].describe().to_dict()
        
        return jsonify({
            'status': 'success',
            'card_recommendation': card_prediction,
            'risk_level': risk_prediction,
            'city_stats': city_stats,
            'message': 'Analysis completed successfully'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/get_stats', methods=['GET'])
def get_stats():
    try:
        stats = {
            'total_transactions': len(df),
            'total_amount': df['Amount'].sum(),
            'avg_transaction': df['Amount'].mean(),
            'top_cities': df['City'].value_counts().head(5).to_dict(),
            'card_distribution': df['Card Type'].value_counts().to_dict()
        }
        return jsonify({'status': 'success', 'data': stats})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)