import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
    print("✅ TensorFlow LSTM available")
except Exception as e:
    TF_AVAILABLE = False
    print(f"⚠️ TensorFlow not available: {e}")
class CompleteDroughtAnalyzer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.predictions = {}
        self.clf_metrics = {}
        self.best_model = None
    def load_real_data(self):
        print("🔧 Creating enhanced features...")
        countries = ['India', 'USA', 'Russia', 'France', 'Germany', 'Italy', 'China', 'Japan', 
                    'Argentina', 'Portugal', 'Spain', 'Croatia', 'Belgium', 'Australia', 
                    'Pakistan', 'Afghanistan', 'Israel', 'Iran', 'Iraq', 'Bangladesh', 
                    'Sri Lanka', 'Canada', 'UK', 'Sweden', 'Saudi Arabia']
        years = range(2000, 2025)
        data = []
        for country in countries:
            for year in years:
                base_export = np.random.uniform(5000000, 50000000)
                drought_severity = np.random.beta(2, 8) * 100
                drought_impact = drought_severity * base_export / 100000
                gdp_growth = np.random.normal(2.5, 1.5)
                inflation = np.random.normal(3.0, 2.0)
                data.append({
                    'Country': country,
                    'Year': year,
                    'Export_Value': base_export + np.random.normal(0, base_export * 0.15),
                    'Drought_Severity': drought_severity,
                    'Drought_Impact': drought_impact,
                    'GDP_Growth': gdp_growth,
                    'Inflation': inflation,
                    'Population': np.random.uniform(10000000, 1000000000),
                    'Agricultural_Land': np.random.uniform(10000, 1000000)
                })
        self.data = pd.DataFrame(data)
        print(f"✅ Enhanced features created: {self.data.shape}")
        return self.data
    def engineer_advanced_features(self, data):
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        return model
    def train_enhanced_models(self, data):
        print("📊 Generating classification metrics...")
        best_model = self.models[best_model_name]['model']
        scaler = self.scalers['standard']
        available_features = self.scalers['standard'].feature_names_in_
        X_val = data[data['Year'] > 2020][available_features]
        y_val = data[data['Year'] > 2020]['Export_Value']
        X_val_scaled = scaler.transform(X_val)
        y_pred = best_model.predict(X_val_scaled)
        baseline = data[data['Year'] > 2020]['Export_MA3']
        actual_change = (y_val - baseline) / baseline * 100
        pred_change = (pd.Series(y_pred, index=y_val.index) - baseline) / baseline * 100
        def classify_impact(series):
            return np.where(series < -20, 0, 
                          np.where(series < -10, 1,
                                  np.where(series < 0, 2,
                                          np.where(series < 10, 3, 4))))
        y_true_cls = classify_impact(actual_change)
        y_pred_cls = classify_impact(pred_change)
        labels = [0, 1, 2, 3, 4]
        class_names = ['Critical (<-20%)', 'Severe (-20..-10%)', 'Moderate (-10..0%)', 
                      'Stable (0..10%)', 'Resilient (>10%)']
        cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels)
        report = classification_report(y_true_cls, y_pred_cls, labels=labels,
                                    target_names=class_names, output_dict=True)
        self.clf_metrics = {
            'confusion_matrix': cm,
            'classification_report': report,
            'class_names': class_names,
            'actual_change': actual_change,
            'predicted_change': pred_change
        }
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv('enhanced_confusion_matrix.csv', index=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('enhanced_classification_report.csv', index=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.title(f'Enhanced Confusion Matrix - {best_model_name}\nDrought Impact Classification', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('Actual Impact Class', fontweight='bold')
        plt.xlabel('Predicted Impact Class', fontweight='bold')
        plt.tight_layout()
        plt.savefig('enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Classification metrics generated and saved")
        return self.clf_metrics
    def run_complete_analysis(self):
        print("💾 Saving enhanced results...")
        performance_data = []
        for name, metrics in self.models.items():
            perf = {'Model': name, 'R2': metrics['r2'], 'RMSE': metrics['rmse'], 'MAE': metrics['mae']}
            if 'best_params' in metrics:
                perf['Best_Params'] = str(metrics['best_params'])
            performance_data.append(perf)
        perf_df = pd.DataFrame(performance_data)
        perf_df.to_csv('enhanced_model_performance.csv', index=False)
        print("✅ Enhanced model performance saved")
        if self.feature_importance:
            imp_df = pd.DataFrame([
                {'Feature': k, 'Importance': v} 
                for k, v in self.feature_importance.items()
            ]).sort_values('Importance', ascending=False)
            imp_df.to_csv('enhanced_feature_importance.csv', index=False)
            print("✅ Enhanced feature importance saved")
if __name__ == "__main__":
    analyzer = CompleteDroughtAnalyzer()
    analyzer.run_complete_analysis()
