import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
class FastDroughtAnalyzer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.predictions = {}
    def load_data(self):
        print("🧹 Cleaning data...")
        countries = ['India', 'USA', 'Russia', 'France', 'Germany', 'Italy', 'China', 'Japan', 
                    'Argentina', 'Portugal', 'Spain', 'Croatia', 'Belgium', 'Australia', 
                    'Pakistan', 'Afghanistan', 'Israel', 'Iran', 'Iraq', 'Bangladesh', 
                    'Sri Lanka', 'Canada', 'UK', 'Sweden', 'Saudi Arabia']
        years = range(2000, 2025)
        data = []
        for country in countries:
            for year in years:
                base_export = np.random.uniform(1000000, 10000000)
                drought_severity = np.random.beta(2, 5) * 100
                drought_impact = drought_severity * base_export / 10000
                data.append({
                    'Country': country,
                    'Year': year,
                    'Export_Value': base_export + np.random.normal(0, base_export * 0.1),
                    'Drought_Severity': drought_severity,
                    'Drought_Impact': drought_impact
                })
        self.clean_data = pd.DataFrame(data)
        print(f"✅ Clean data created: {len(self.clean_data)} records")
        return self.clean_data
    def create_features(self, data):
        model = Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_features,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )
        return model
    def train_models(self, data):
        print("🔮 Predicting 2030...")
        best_model = self.models[best_model_name]['model']
        scaler = self.scalers['standard']
        scenario_data = []
        for country in data['Country'].unique():
            country_data = data[data['Country'] == country].iloc[-1:].copy()
            for year in [2027, 2028, 2029]:
                scenario_row = country_data.copy()
                scenario_row['Year'] = year
                scenario_row['Drought_Severity'] = np.random.uniform(80, 100)
                scenario_row['Drought_Impact'] = scenario_row['Drought_Severity'] * scenario_row['Export_Lag1'] / 10000
                scenario_data.append(scenario_row)
            scenario_row = country_data.copy()
            scenario_row['Year'] = 2030
            scenario_row['Drought_Severity'] = 40
            scenario_row['Drought_Impact'] = 32
            scenario_data.append(scenario_row)
        scenario_df = pd.DataFrame(scenario_data)
        scenario_df = self.update_scenario_features(scenario_df, data)
        features = ['Drought_Severity', 'Drought_Impact', 'Export_Lag1', 'Export_Lag2', 
                   'Drought_Lag1', 'Export_MA3', 'Drought_MA3', 'Drought_Export_Interaction', 
                   'Country_Vulnerability']
        available_features = [f for f in features if f in scenario_df.columns]
        X_scenario = scenario_df[available_features]
        X_scenario_scaled = scaler.transform(X_scenario)
        if best_model_name == 'Advanced TensorFlow':
            scenario_df['Predicted_Export_2030'] = best_model.predict(X_scenario_scaled).flatten()
        else:
            scenario_df['Predicted_Export_2030'] = best_model.predict(X_scenario_scaled)
        baseline_exports = data.groupby('Country')['Export_Value'].mean()
        scenario_df = scenario_df.merge(baseline_exports.reset_index(), on='Country', suffixes=('', '_Baseline'))
        scenario_df['Export_Change_Percent'] = (
            (scenario_df['Predicted_Export_2030'] - scenario_df['Export_Value_Baseline']) / 
            scenario_df['Export_Value_Baseline'] * 100
        )
        self.predictions['2030_scenario'] = scenario_df
        print(f"✅ 2030 predictions completed for {len(scenario_df)} scenarios")
        return scenario_df
    def update_scenario_features(self, scenario_df, historical_data):
        print("💡 Generating insights...")
        if '2030_scenario' not in self.predictions:
            print("❌ No predictions available.")
            return
        scenario_df = self.predictions['2030_scenario']
        vulnerability_2030 = scenario_df[scenario_df['Year'] == 2030].copy()
        vulnerability_2030 = vulnerability_2030.sort_values('Export_Change_Percent')
        print("\n" + "="*80)
        print("🌾 TOP 10 COUNTRIES MOST VULNERABLE TO DROUGHT IMPACT BY 2030")
        print("="*80)
        for i, (_, row) in enumerate(vulnerability_2030.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['Country']:<15} | Export Change: {row['Export_Change_Percent']:6.1f}% | "
                  f"Predicted Export: {row['Predicted_Export_2030']:8.0f}")
        total_baseline = vulnerability_2030['Export_Value_Baseline'].sum()
        total_predicted = vulnerability_2030['Predicted_Export_2030'].sum()
        total_impact = ((total_predicted - total_baseline) / total_baseline) * 100
        print(f"\n💰 TOTAL ECONOMIC IMPACT: {total_impact:.1f}% change in agricultural exports")
        print(f"   Baseline: ${total_baseline:,.0f}")
        print(f"   Predicted: ${total_predicted:,.0f}")
        return vulnerability_2030
    def create_visualizations(self):
        print("💾 Saving results...")
        if '2030_scenario' in self.predictions:
            self.predictions['2030_scenario'].to_csv('fast_drought_predictions_2030.csv', index=False)
            print("✅ Predictions saved to 'fast_drought_predictions_2030.csv'")
        model_performance = []
        for name, metrics in self.models.items():
            model_performance.append({
                'Model': name,
                'R2_Score': metrics['r2'],
                'RMSE': metrics['rmse']
            })
        performance_df = pd.DataFrame(model_performance)
        performance_df.to_csv('fast_model_performance.csv', index=False)
        print("✅ Model performance saved to 'fast_model_performance.csv'")
        if self.feature_importance:
            importance_df = pd.DataFrame([
                {'Feature': k, 'Importance': v} 
                for k, v in self.feature_importance.items()
            ]).sort_values('Importance', ascending=False)
            importance_df.to_csv('fast_feature_importance.csv', index=False)
            print("✅ Feature importance saved to 'fast_feature_importance.csv'")
    def run_fast_analysis(self):
