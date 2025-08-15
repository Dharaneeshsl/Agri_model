import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.regularizers import l1_l2
    TF_AVAILABLE = True
    print("✅ TensorFlow loaded successfully!")
except Exception as _tf_err:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(_tf_err)
    print(f"⚠️ TensorFlow not available: {_tf_err}")
class AdvancedDroughtAnalyzer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.predictions = {}
        self.clf_outputs = {}
        self.advanced_features = {}
    def load_data(self):
        print("🧹 Creating enhanced synthetic data...")
        countries = ['India', 'USA', 'Russia', 'France', 'Germany', 'Italy', 'China', 'Japan', 
                    'Argentina', 'Portugal', 'Spain', 'Croatia', 'Belgium', 'Australia', 
                    'Pakistan', 'Afghanistan', 'Israel', 'Iran', 'Iraq', 'Bangladesh', 
                    'Sri Lanka', 'Canada', 'UK', 'Sweden', 'Saudi Arabia']
        years = range(2000, 2025)
        data = []
        for country in countries:
            base_export = np.random.uniform(2000000, 15000000)
            climate_zone = np.random.choice(['Tropical', 'Temperate', 'Arid'])
            development_level = np.random.choice(['Developed', 'Developing', 'Emerging'])
            for year in years:
                drought_trend = 1 + (year - 2000) * 0.02
                drought_severity = np.random.beta(2, 5) * 100 * drought_trend
                economic_cycle = 1 + 0.1 * np.sin((year - 2000) * 0.5)
                export_value = base_export * economic_cycle * (1 + np.random.normal(0, 0.15))
                drought_impact = drought_severity * export_value / 15000
                data.append({
                    'Country': country,
                    'Year': year,
                    'Export_Value': max(export_value, 100000),
                    'Drought_Severity': min(drought_severity, 100),
                    'Drought_Impact': drought_impact,
                    'Climate_Zone': climate_zone,
                    'Development_Level': development_level
                })
        self.clean_data = pd.DataFrame(data)
        print(f"✅ Enhanced synthetic data created: {len(self.clean_data)} records")
        return self.clean_data
    def create_advanced_features(self, data):
        model = models.Sequential([
            layers.Input(shape=(input_features,)),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            layers.Dropout(0.25),
            layers.Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001, weight_decay=1e-5),
            loss='huber',
            metrics=['mae', 'mse']
        )
        return model
    def train_advanced_models(self, data):
        try:
            print("📊 Generating advanced classification metrics...")
            best_model = self.models[best_model_name]['model']
            y_pred = best_model.predict(X_val_scaled)
            ref = val_data['Export_MA3'].replace(0, np.nan).fillna(val_data['Export_MA3'].mean())
            actual_change = (y_val - ref) / ref * 100.0
            pred_change = (pd.Series(y_pred, index=y_val.index) - ref) / ref * 100.0
            def to_advanced_class(series):
                return np.where(series < -20.0, 0, 
                       np.where(series < -10.0, 1,
                       np.where(series < 0.0, 2,
                       np.where(series < 10.0, 3, 4))))
            y_true_cls = to_advanced_class(actual_change)
            y_pred_cls = to_advanced_class(pred_change)
            labels = [0, 1, 2, 3, 4]
            class_names = ['Critical (<-20%)', 'Severe (-20..-10%)', 'Moderate (-10..0%)', 'Stable (0..10%)', 'Growth (>10%)']
            cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels)
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            cm_df.to_csv('advanced_confusion_matrix.csv', index=True)
            report = classification_report(y_true_cls, y_pred_cls, labels=labels, target_names=class_names)
            with open('advanced_classification_report.txt', 'w') as f:
                f.write(report)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
            plt.title(f'Advanced Confusion Matrix - {best_model_name}\n5-Class Classification', fontweight='bold', fontsize=14)
            plt.ylabel('Actual Class', fontweight='bold')
            plt.xlabel('Predicted Class', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('advanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            self.clf_outputs = {
                'confusion_matrix': cm,
                'classification_report': report,
                'actual_classes': y_true_cls,
                'predicted_classes': y_pred_cls,
                'class_names': class_names
            }
            print("✅ Advanced classification metrics generated and saved!")
        except Exception as e:
            print(f"⚠️ Classification metrics generation failed: {e}")
    def predict_2030(self, data, best_model_name):
        for country in scenario_df['Country'].unique():
            country_scenario = scenario_df[scenario_df['Country'] == country].copy()
            country_historical = historical_data[historical_data['Country'] == country]
            for idx, row in country_scenario.iterrows():
                if row['Year'] >= 2027:
                    if row['Year'] == 2027:
                        prev_years = country_historical[country_historical['Year'] >= 2024]['Export_Value'].tail(3).tolist()
                        prev_years.append(row['Export_Lag1'])
                        prev_years.append(row['Export_Lag2'])
                    elif row['Year'] == 2028:
                        prev_years = country_historical[country_historical['Year'] >= 2025]['Export_Value'].tail(2).tolist()
                        prev_years.extend([row['Export_Lag1'], row['Export_Lag2'], row['Export_Lag3'])
                    elif row['Year'] == 2029:
                        prev_years = [row['Export_Lag1'], row['Export_Lag2'], row['Export_Lag3']]
                        prev_years.append(country_historical[country_historical['Year'] == 2026]['Export_Value'].iloc[0] if len(country_historical[country_historical['Year'] == 2026]) > 0 else 0)
                    else:
                        prev_years = [row['Export_Lag1'], row['Export_Lag2'], row['Export_Lag3']]
                        prev_years.append(country_historical[country_historical['Year'] == 2027]['Export_Value'].iloc[0] if len(country_historical[country_historical['Year'] == 2027]) > 0 else 0)
                    scenario_df.loc[idx, 'Export_MA3'] = np.mean(prev_years[:3])
                    scenario_df.loc[idx, 'Export_MA5'] = np.mean(prev_years[:5])
                    scenario_df.loc[idx, 'Export_Volatility'] = np.std(prev_years[:3])
                    if row['Year'] == 2027:
                        prev_drought = country_historical[country_historical['Year'] >= 2024]['Drought_Severity'].tail(3).tolist()
                        prev_drought.append(row['Drought_Lag1'])
                        prev_drought.append(row['Drought_Lag2'])
                    elif row['Year'] == 2028:
                        prev_drought = country_historical[country_historical['Year'] >= 2025]['Drought_Severity'].tail(2).tolist()
                        prev_drought.extend([row['Drought_Lag1'], 88, 92])
                    elif row['Year'] == 2029:
                        prev_drought = [row['Drought_Lag1'], 92, 95, 98]
                    else:
                        prev_drought = [row['Drought_Lag1'], 95, 98, 40]
                    scenario_df.loc[idx, 'Drought_MA3'] = np.mean(prev_drought[:3])
                    scenario_df.loc[idx, 'Drought_MA5'] = np.mean(prev_drought[:5])
                    scenario_df.loc[idx, 'Drought_Volatility'] = np.std(prev_drought[:3])
                    scenario_df.loc[idx, 'Drought_Export_Interaction'] = row['Drought_Severity'] * row['Export_Lag1'] / 10000
                    scenario_df.loc[idx, 'Drought_Export_Ratio'] = row['Drought_Severity'] / (row['Export_Lag1'] + 1)
                    scenario_df.loc[idx, 'Year_Trend'] = row['Year'] - 2000
                    scenario_df.loc[idx, 'Decade'] = (row['Year'] // 10) * 10
                    if row['Year'] > 2027:
                        prev_growth = scenario_df.loc[scenario_df['Country'] == country, 'Export_Growth_Rate'].iloc[:-1].tail(1).iloc[0] if len(scenario_df[scenario_df['Country'] == country]) > 1 else 0
                        scenario_df.loc[idx, 'Export_Growth_Rate'] = prev_growth
                        scenario_df.loc[idx, 'Export_Growth_Acceleration'] = 0
        return scenario_df
    def generate_insights(self):
        print("📊 Creating comprehensive visualizations...")
        if '2030_scenario' not in self.predictions:
            print("❌ No predictions available.")
            return
        scenario_df = self.predictions['2030_scenario']
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
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
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['r2'] for name in model_names]
        rmse_scores = [self.models[name]['rmse'] for name in model_names]
        x = np.arange(len(model_names))
        width = 0.35
        bars1 = ax2.bar(x - width/2, r2_scores, width, label='R² Score', alpha=0.8, color='lightblue')
        bars2 = ax2.bar(x + width/2, rmse_scores, width, label='RMSE', alpha=0.8, color='lightcoral')
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Score')
        ax2.set_title('Model Performance Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.legend()
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1f}', ha='center', va='bottom')
        ax3 = fig.add_subplot(gs[1, :2])
        comparison_data = vulnerability_2030.head(10)
        x_pos = np.arange(len(comparison_data))
        width = 0.35
        ax3.bar(x_pos - width/2, comparison_data['Export_Value_Baseline'], 
               width, label='Baseline', alpha=0.8, color='skyblue')
        ax3.bar(x_pos + width/2, comparison_data['Predicted_Export_2030'], 
               width, label='Predicted 2030', alpha=0.8, color='lightcoral')
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
        impact_2030 = scenario_df[scenario_df['Year'] == 2030]['Export_Change_Percent']
        ax5.hist(impact_2030, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax5.axvline(impact_2030.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {impact_2030.mean():.1f}%')
        ax5.axvline(impact_2030.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {impact_2030.median():.1f}%')
        ax5.set_xlabel('Export Change (%)')
        ax5.set_ylabel('Number of Countries')
        ax5.set_title('Distribution of Drought Impact on Agricultural Exports (2030)', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.suptitle('Advanced Drought Impact Analysis: 3-Year Drought Scenario (2027-2029)', 
                     fontsize=18, fontweight='bold')
        plt.savefig('comprehensive_drought_impact_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Comprehensive visualizations saved as 'comprehensive_drought_impact_analysis.png'")
        return fig
    def save_comprehensive_results(self):
        print("📋 Generating comprehensive report...")
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE DROUGHT IMPACT ANALYSIS - PROBLEM 3")
        report.append("="*80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("MODEL PERFORMANCE SUMMARY:")
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
        with open('comprehensive_drought_impact_report.txt', 'w') as f:
            f.write('\n'.join(report))
        print("✅ Comprehensive report saved to 'comprehensive_drought_impact_report.txt'")
    def run_comprehensive_analysis(self):
