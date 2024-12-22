# Credit Card Transaction Analysis Project

## Overview
This project analyzes credit card transaction data to provide insights and build predictive models for better financial decision-making. The analysis includes data preprocessing, exploratory data analysis (EDA), visualization, and machine learning model development.
This dataset is available in [KAGGLE](https://www.kaggle.com/datasets/thedevastator/analyzing-credit-card-spending-habits-in-india) Thanks to the original author of the dataset.

## Features
- Data preprocessing and cleaning
- Comprehensive exploratory data analysis
- Statistical analysis of transaction patterns
- Multiple machine learning models for different business objectives:
  1. Card Recommendation System
  2. Risk Assessment Model
  3. Geographical Analysis
  4. Customer Experience Enhancement

## Dependencies
The project requires the following Python libraries:
- scikit-learn
- seaborn
- matplotlib
- xgboost
- pandas
- numpy
- scipy

Install dependencies using:
```bash
pip install scikit-learn seaborn matplotlib xgboost
```

## Data Analysis Components

### 1. Data Preprocessing
- Date formatting and standardization
- Handling outliers in transaction amounts
- Feature scaling using MinMax and Standard scalers
- Categorical variable encoding

### 2. Exploratory Data Analysis
- Transaction patterns by city
- Temporal analysis of spending
- Card type distribution
- Gender-based analysis
- Expense type analysis
- Amount distribution and outlier detection

### 3. Statistical Analysis
- ANOVA tests for card types
- T-tests for gender-based spending
- Correlation analysis
- Multi-dimensional analysis of spending patterns

### 4. Machine Learning Models

#### Card Recommendation Model
- Uses XGBoost classifier
- Features: Gender, Amount, Expense Type, Month, City
- Predicts suitable card types for customers

#### Risk Assessment Model
- Random Forest Classifier
- Analyzes spending patterns and transaction ratios
- Categorizes transactions into risk levels (Low, Medium, High)

#### Geographical Analysis Model
- K-means clustering for city segmentation
- Analyzes city-level metrics and demographics
- Helps in geographical expansion strategies

#### Customer Experience Model
- Customer segmentation using PCA and K-means
- Analyzes transaction frequency and spending diversity
- Helps in personalizing customer experience

## Usage
The script processes credit card transaction data from 'creditcard.csv' and performs various analyses. The main components can be run sequentially to generate insights and predictions.

## Results
The analysis provides:
- Visualizations of spending patterns
- Statistical insights into customer behavior
- Predictive models for card recommendations
- Risk assessment capabilities
- Geographical expansion insights
- Customer segmentation for personalized services

## Future Improvements
- Real-time transaction processing
- Integration with web interface
- Enhanced visualization capabilities
- API development for model deployment

NOTE: This is a prototype project done for students understanding on Datasets, Data Cleaning, ML Alogorithms, Visualizations and etc, it is under development for realtime 
