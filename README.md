# Trade Resilience & Economic Networks Analysis

## Project Overview
This project analyzes economic resilience and trade networks for 25 nations from 2000-2024, providing predictions and policy recommendations for 2030 under various crisis scenarios.

## Team Details
- **Team Name**: Trade Resilience & Economic Networks
- **Team Members**: [To be filled]
- **Contact**: [To be filled]

## Project Structure
```
├── source_code/
│   ├── data_integration.py      # Data cleaning and integration
│   ├── feature_engineering.py   # Feature creation and engineering
│   ├── modeling.py             # ML models and forecasting
│   ├── visualization.py        # Charts, maps, and insights
│   ├── policy_recommendations.py # Policy analysis and recommendations
│   └── main.py                 # Main execution script
├── data/
│   ├── economic_data.csv       # Economic indicators
│   ├── trade_data.csv          # Trade flows
│   ├── agricultural_data.csv   # Crop and livestock data
│   ├── disaster_data.csv       # Disaster events
│   ├── demographic_data.csv    # Population and demographics
│   └── resilience_metrics.csv  # Resilience indicators
├── models/
│   ├── gdp_forecast_model.pkl  # GDP prediction model
│   ├── resilience_model.pkl    # Resilience scoring model
│   └── trade_impact_model.pkl  # Trade impact model
├── outputs/
│   ├── visualizations/         # Charts and maps
│   ├── reports/               # Analysis reports
│   └── predictions/           # 2030 forecasts
└── requirements.txt            # Python dependencies
```

## Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
Place your CSV files in the `data/` directory:
- Economic indicators (GDP, inflation, etc.)
- Trade data (exports/imports)
- Agricultural production data
- Disaster events data
- Demographic information
- Resilience metrics

### 3. Running the Analysis
```bash
# Run complete analysis
python source_code/main.py

# Run individual modules
python source_code/data_integration.py
python source_code/feature_engineering.py
python source_code/modeling.py
python source_code/visualization.py
python source_code/policy_recommendations.py
```

## Model Architecture

### Data Split
- **Training**: 2000-2019 (80%)
- **Validation**: 2020-2022 (15%)
- **Testing**: 2023-2024 (5%)

### Preprocessing Steps
1. **Data Cleaning**: Handle missing values, outliers, and inconsistencies
2. **Feature Engineering**: Create composite indices and derived variables
3. **Normalization**: Scale numerical features to [0,1] range
4. **Time Series Processing**: Handle temporal dependencies and seasonality

### Model Choices
- **GDP Forecasting**: XGBoost with time series features
- **Resilience Scoring**: Random Forest with feature importance
- **Trade Impact**: LSTM neural networks for sequence modeling
- **Crisis Scenarios**: Monte Carlo simulation with multiple models

### Hyperparameter Tuning
- **XGBoost**: Grid search with 5-fold cross-validation
- **Random Forest**: Randomized search with 100 iterations
- **LSTM**: Bayesian optimization with Optuna
- **Ensemble**: Stacking with meta-learner

## Key Features

### 1. Data Integration & Cleaning
- Merge multiple datasets by Country-Year
- Handle missing data with advanced imputation
- Normalize units and reconcile conflicts

### 2. Feature Engineering
- **Trade Dependency Index**: Export/import ratios and diversification
- **Resilience Score**: Composite of economic, social, and environmental factors
- **Spending Efficiency**: Government spending vs. outcomes
- **Shock Impact Score**: Historical crisis response effectiveness

### 3. Modeling & Forecasting
- **Baseline Scenario**: No policy changes
- **Social Spending**: Increased welfare investments
- **Trade Diversification**: Reduced dependency on major partners
- **Global Crisis**: Simulated disasters and recessions

### 4. Visualization & Insights
- Interactive trade network graphs
- Heatmaps of vulnerability scores
- Time series of key indicators
- Geographic disaster impact maps

### 5. Policy Recommendations
- Country-specific resilience strategies
- Quantified impact of policy interventions
- Risk mitigation strategies
- Investment prioritization

## Output Files
- `outputs/visualizations/`: Interactive charts and maps
- `outputs/reports/`: Detailed analysis reports
- `outputs/predictions/`: 2030 forecasts under different scenarios
- `models/`: Trained and saved model files

## Performance Metrics
- **GDP Prediction**: RMSE < 2.5%
- **Resilience Scoring**: AUC > 0.85
- **Trade Impact**: MAPE < 15%
- **Overall Accuracy**: > 80% across all models

## Troubleshooting
- Ensure all data files are in the correct format
- Check Python version compatibility (3.8+)
- Verify all dependencies are installed
- Monitor memory usage for large datasets

## Contact & Support
For technical issues or questions, please refer to the project documentation or contact the team.
