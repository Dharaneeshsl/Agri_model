import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, classification_report
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
class AdvancedDroughtAnalyzer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.predictions = {}
        self.evaluation_metrics = {}
    def load_and_prepare_data(self):
        print("🧹 Cleaning FAOSTAT data...")
        export_data = self.faostat_data[
            (self.faostat_data['Element'].str.contains('Export', na=False)) &
            (self.faostat_data['Domain'] == 'Crops and livestock products')
        ].copy()
        export_data['Country'] = export_data['Area'].str.strip()
        export_data['Year'] = pd.to_numeric(export_data['Year'], errors='coerce')
        export_data['Value'] = pd.to_numeric(export_data['Value'], errors='coerce')
        focus_countries = [
            'India', 'USA', 'Russia', 'France', 'Germany', 'Italy', 'China', 'Japan', 
            'Argentina', 'Portugal', 'Spain', 'Croatia', 'Belgium', 'Australia', 
            'Pakistan', 'Afghanistan', 'Israel', 'Iran', 'Iraq', 'Bangladesh', 
            'Sri Lanka', 'Canada', 'UK', 'Sweden', 'Saudi Arabia'
        ]
        export_data = export_data[
            (export_data['Year'].between(2000, 2024)) &
            (export_data['Country'].isin(focus_countries))
        ]
        export_pivot = export_data.pivot_table(
            index=['Country', 'Year'],
            columns='Element',
            values='Value',
            aggfunc='sum'
        ).reset_index()
        if 'Export quantity' in export_pivot.columns and 'Export value' in export_pivot.columns:
            export_pivot.columns = ['Country', 'Year', 'Export_Quantity', 'Export_Value']
        else:
            export_pivot.columns = ['Country', 'Year', 'Export_Value']
            export_pivot['Export_Quantity'] = export_pivot['Export_Value'] / 1000
        print(f"✅ FAOSTAT data cleaned: {len(export_pivot)} country-year records")
        return export_pivot
    def clean_trade_data(self):
        print("🌾 Creating drought scenarios...")
        base_vulnerability = {
            'India': 0.7, 'USA': 0.5, 'Argentina': 0.8, 'Australia': 0.6,
            'Pakistan': 0.8, 'Afghanistan': 0.9, 'Iran': 0.8, 'Iraq': 0.7,
            'Bangladesh': 0.6, 'Sri Lanka': 0.5, 'Saudi Arabia': 0.9
        }
        np.random.seed(42)
        years = range(2000, 2025)
        countries = [
            'India', 'USA', 'Russia', 'France', 'Germany', 'Italy', 'China', 'Japan', 
            'Argentina', 'Portugal', 'Spain', 'Croatia', 'Belgium', 'Australia', 
            'Pakistan', 'Afghanistan', 'Israel', 'Iran', 'Iraq', 'Bangladesh', 
            'Sri Lanka', 'Canada', 'UK', 'Sweden', 'Saudi Arabia'
        ]
        drought_data = []
        for country in countries:
            for year in years:
                base_vulnerability_val = base_vulnerability.get(country, 0.4)
                drought_severity = np.random.beta(2, 5) * 100 * base_vulnerability_val
                if year > 2000 and drought_data:
                    prev_severity = drought_data[-1]['Drought_Severity']
                    drought_severity = 0.3 * prev_severity + 0.7 * drought_severity
                drought_data.append({
                    'Country': country,
                    'Year': year,
                    'Drought_Severity': drought_severity,
                    'Drought_Impact_Score': drought_severity * base_vulnerability_val
                })
        drought_df = pd.DataFrame(drought_data)
        print(f"✅ Drought scenarios created: {len(drought_df)} records")
        return drought_df
    def engineer_features(self, export_data, trade_data, drought_data):
        print("🏗️ Building Advanced TensorFlow Model...")
        model = Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_features,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        return model
    def train_models(self, data):
        print("🔍 Calculating TensorFlow feature importance...")
        base_score = self.models['Advanced TensorFlow']['model'].evaluate(X_val, y_val, verbose=0)[0]
        importance_scores = {}
        for i, feature in enumerate(features):
            X_val_shuffled = X_val.copy()
            np.random.shuffle(X_val_shuffled[:, i])
            new_score = self.models['Advanced TensorFlow']['model'].evaluate(X_val_shuffled, y_val, verbose=0)[0]
            importance_scores[feature] = new_score - base_score
        max_importance = max(importance_scores.values())
        if max_importance > 0:
            importance_scores = {k: v/max_importance for k, v in importance_scores.items()}
        return importance_scores
    def predict_drought_impact_2030(self, data, best_model_name):
        for country in scenario_df['Country'].unique():
            country_scenario = scenario_df[scenario_df['Country'] == country].copy()
            country_historical = historical_data[historical_data['Country'] == country]
            for idx, row in country_scenario.iterrows():
                if row['Year'] >= 2027:
                    if row['Year'] == 2027:
                        prev_years = country_historical[country_historical['Year'] >= 2024]['Export_Value'].tail(2).tolist()
                        prev_years.append(row['Export_Value_Lag1'])
                    elif row['Year'] == 2028:
                        prev_years = country_historical[country_historical['Year'] >= 2025]['Export_Value'].tail(1).tolist()
                        prev_years.extend([row['Export_Value_Lag1'], row['Export_Value_Lag2']])
                    elif row['Year'] == 2029:
                        prev_years = [row['Export_Value_Lag1'], row['Export_Value_Lag2']]
                        prev_years.append(country_historical[country_historical['Year'] == 2026]['Export_Value'].iloc[0] if len(country_historical[country_historical['Year'] == 2026]) > 0 else 0)
                    else:
                        prev_years = [row['Export_Value_Lag1'], row['Export_Value_Lag2']]
                        prev_years.append(country_historical[country_historical['Year'] == 2027]['Export_Value'].iloc[0] if len(country_historical[country_historical['Year'] == 2027]) > 0 else 0)
                    scenario_df.loc[idx, 'Export_Value_MA3'] = np.mean(prev_years)
                    if row['Year'] == 2027:
                        prev_drought = country_historical[country_historical['Year'] >= 2024]['Drought_Severity'].tail(2).tolist()
                        prev_drought.append(row['Drought_Severity_Lag1'])
                    elif row['Year'] == 2028:
                        prev_drought = country_historical[country_historical['Year'] >= 2025]['Drought_Severity'].tail(1).tolist()
                        prev_drought.extend([row['Drought_Severity_Lag1'], 85])
                    elif row['Year'] == 2029:
                        prev_drought = [row['Drought_Severity_Lag1'], 85, 90]
                    else:
                        prev_drought = [row['Drought_Severity_Lag1'], 90, 95]
                    scenario_df.loc[idx, 'Drought_Severity_MA3'] = np.mean(prev_drought)
                    scenario_df.loc[idx, 'Cumulative_Drought_Impact'] = sum(prev_drought)
                    scenario_df.loc[idx, 'Drought_Export_Impact'] = row['Drought_Severity'] * row['Export_Value_Lag1'] / 1000
        return scenario_df
    def generate_comprehensive_evaluation(self):
        print("📊 Creating advanced visualizations...")
        if '2030_scenario' not in self.predictions:
            print("❌ No predictions available. Run prediction first.")
            return
        scenario_df = self.predictions['2030_scenario']
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :2])
        vulnerability_2030 = scenario_df[scenario_df['Year'] == 2030].copy()
        vulnerability_2030 = vulnerability_2030.sort_values('Export_Change_Percent').head(15)
        bars = ax1.barh(range(len(vulnerability_2030)), vulnerability_2030['Export_Change_Percent'])
        ax1.set_yticks(range(len(vulnerability_2030)))
        ax1.set_yticklabels(vulnerability_2030['Country'])
        ax1.set_xlabel('Export Change (%)')
        ax1.set_title('Top 15 Most Vulnerable Countries (2030)', fontweight='bold')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        for i, bar in enumerate(bars):
            if vulnerability_2030.iloc[i]['Export_Change_Percent'] < -20:
                bar.set_color('darkred')
            elif vulnerability_2030.iloc[i]['Export_Change_Percent'] < -10:
                bar.set_color('red')
            elif vulnerability_2030.iloc[i]['Export_Change_Percent'] < 0:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        ax2 = fig.add_subplot(gs[0, 2])
        drought_timeline = scenario_df[scenario_df['Country'].isin(vulnerability_2030['Country'].head(8))]
        for country in drought_timeline['Country'].unique():
            country_data = drought_timeline[drought_timeline['Country'] == country]
            ax2.plot(country_data['Year'], country_data['Drought_Severity'], 
                     marker='o', label=country, linewidth=2)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Drought Severity (0-100)')
        ax2.set_title('Drought Severity Timeline (2027-2029)', fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax3 = fig.add_subplot(gs[1, :2])
        comparison_data = vulnerability_2030.head(10)
        x_pos = np.arange(len(comparison_data))
        width = 0.35
        ax3.bar(x_pos - width/2, comparison_data['Export_Value_Baseline'], 
                width, label='Baseline (Normal)', alpha=0.8, color='skyblue')
        ax3.bar(x_pos + width/2, comparison_data['Predicted_Export_Value_2030'], 
                width, label='Predicted (Post-Drought)', alpha=0.8, color='lightcoral')
        ax3.set_xlabel('Countries')
        ax3.set_ylabel('Export Value')
        ax3.set_title('Export Value Comparison: Baseline vs Predicted 2030', fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(comparison_data['Country'], rotation=45, ha='right')
        ax3.legend()
        ax4 = fig.add_subplot(gs[1, 2])
        if self.feature_importance:
            features = list(self.feature_importance.keys())
            importance = list(self.feature_importance.values())
            sorted_idx = np.argsort(importance)[::-1]
            features = [features[i] for i in sorted_idx]
            importance = [importance[i] for i in sorted_idx]
            bars = ax4.barh(range(len(features)), importance)
            ax4.set_yticks(range(len(features)))
            ax4.set_yticklabels(features)
            ax4.set_xlabel('Feature Importance')
            ax4.set_title('Model Feature Importance', fontweight='bold')
            for i, bar in enumerate(bars):
                if i < 3:
                    bar.set_color('darkgreen')
                elif i < 6:
                    bar.set_color('green')
                else:
                    bar.set_color('lightgreen')
        ax5 = fig.add_subplot(gs[2, :])
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['r2'] for name in model_names]
        rmse_scores = [self.models[name]['rmse'] for name in model_names]
        x = np.arange(len(model_names))
        width = 0.35
        bars1 = ax5.bar(x - width/2, r2_scores, width, label='R² Score', alpha=0.8, color='lightblue')
        bars2 = ax5.bar(x + width/2, rmse_scores, width, label='RMSE', alpha=0.8, color='lightcoral')
        ax5.set_xlabel('Models')
        ax5.set_ylabel('Score')
        ax5.set_title('Model Performance Comparison', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(model_names, rotation=45, ha='right')
        ax5.legend()
        for bar in bars1:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1f}', ha='center', va='bottom')
        ax6 = fig.add_subplot(gs[3, :])
        impact_2030 = scenario_df[scenario_df['Year'] == 2030]['Export_Change_Percent']
        ax6.hist(impact_2030, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax6.axvline(impact_2030.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {impact_2030.mean():.1f}%')
        ax6.axvline(impact_2030.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {impact_2030.median():.1f}%')
        ax6.set_xlabel('Export Change (%)')
        ax6.set_ylabel('Number of Countries')
        ax6.set_title('Distribution of Drought Impact on Agricultural Exports (2030)', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        plt.suptitle('Advanced Agricultural Drought Impact Analysis: 3-Year Drought Scenario (2027-2029)', 
                     fontsize=18, fontweight='bold')
        plt.savefig('advanced_drought_impact_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Advanced visualizations saved as 'advanced_drought_impact_analysis.png'")
        return fig
    def save_comprehensive_results(self):
        print("📋 Generating comprehensive report...")
        report = []
        report.append("="*80)
        report.append("ADVANCED AGRICULTURAL DROUGHT IMPACT ANALYSIS - PROBLEM 3")
        report.append("="*80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("ADVANCED MODEL PERFORMANCE SUMMARY:")
        report.append("-" * 50)
        for name, metrics in self.models.items():
            report.append(f"{name:<25} | R²: {metrics['r2']:6.4f} | RMSE: {metrics['rmse']:8.2f} | MAE: {metrics['mae']:8.2f}")
        report.append("")
        if '2030_scenario' in self.predictions:
            scenario_df = self.predictions['2030_scenario']
            vulnerability_2030 = scenario_df[scenario_df['Year'] == 2030].copy()
            vulnerability_2030 = vulnerability_2030.sort_values('Export_Change_Percent')
            report.append("KEY FINDINGS:")
            report.append("-" * 40)
            report.append(f"Total countries analyzed: {len(vulnerability_2030)}")
            report.append(f"Countries with >20% export decline: {len(vulnerability_2030[vulnerability_2030['Export_Change_Percent'] < -20])}")
            report.append(f"Countries with >10% export decline: {len(vulnerability_2030[vulnerability_2030['Export_Change_Percent'] < -10])}")
            report.append(f"Countries with <5% export decline: {len(vulnerability_2030[vulnerability_2030['Export_Change_Percent'] > -5])}")
            report.append("")
            report.append("TOP 5 MOST VULNERABLE COUNTRIES:")
            report.append("-" * 40)
            for i, (_, row) in enumerate(vulnerability_2030.head(5).iterrows(), 1):
                report.append(f"{i}. {row['Country']}: {row['Export_Change_Percent']:.1f}% change")
            report.append("")
            report.append("TOP 5 MOST RESILIENT COUNTRIES:")
            report.append("-" * 40)
            for i, (_, row) in enumerate(vulnerability_2030.tail(5).iterrows(), 1):
                report.append(f"{i}. {row['Country']}: {row['Export_Change_Percent']:.1f}% change")
            report.append("")
            report.append("STATISTICAL SUMMARY:")
            report.append("-" * 40)
            report.append(f"Mean export change: {vulnerability_2030['Export_Change_Percent'].mean():.2f}%")
            report.append(f"Median export change: {vulnerability_2030['Export_Change_Percent'].median():.2f}%")
            report.append(f"Standard deviation: {vulnerability_2030['Export_Change_Percent'].std():.2f}%")
            report.append(f"Range: {vulnerability_2030['Export_Change_Percent'].min():.2f}% to {vulnerability_2030['Export_Change_Percent'].max():.2f}%")
        with open('advanced_drought_impact_comprehensive_report.txt', 'w') as f:
            f.write('\n'.join(report))
        print("✅ Comprehensive report saved to 'advanced_drought_impact_comprehensive_report.txt'")
    def run_complete_analysis(self):
