import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class EconomicModeler:
        try:
            data_path = self.data_dir / "engineered_features.csv"
            if data_path.exists():
                df = pd.read_csv(data_path)
                logger.info(f"Loaded feature data: {df.shape}")
                return df
            else:
                logger.info("Feature data not found, running feature engineering...")
                from feature_engineering import FeatureEngineer
                engineer = FeatureEngineer()
                return engineer.engineer_all_features()
        except Exception as e:
            logger.error(f"Error loading feature data: {e}")
            raise
    def prepare_modeling_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Training GDP forecasting model...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        train_pred = xgb_model.predict(X_train)
        val_pred = xgb_model.predict(X_val)
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred)
        }
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        model_info = {
            'model': xgb_model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'model_type': 'XGBoost'
        }
        logger.info(f"GDP Model - Train RMSE: {metrics['train_rmse']:.4f}, Val RMSE: {metrics['val_rmse']:.4f}")
        return model_info
    def train_resilience_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        logger.info("Training trade impact model...")
        sequence_length = 5
        def create_sequences(X, y, seq_length):
            X_seq, y_seq = [], []
            for i in range(seq_length, len(X)):
                X_seq.append(X.iloc[i-seq_length:i].values)
                y_seq.append(y.iloc[i])
            return np.array(X_seq), np.array(y_seq)
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
        if len(X_train_seq) > 0 and len(X_val_seq) > 0:
            lstm_model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='linear')
            ])
            lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            history = lstm_model.fit(
                X_train_seq, y_train_seq,
                epochs=50,
                batch_size=32,
                validation_data=(X_val_seq, y_val_seq),
                verbose=0
            )
            train_pred = lstm_model.predict(X_train_seq).flatten()
            val_pred = lstm_model.predict(X_val_seq).flatten()
            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train_seq, train_pred)),
                'val_rmse': np.sqrt(mean_squared_error(y_val_seq, val_pred)),
                'train_r2': r2_score(y_train_seq, train_pred),
                'val_r2': r2_score(y_val_seq, val_pred)
            }
            model_info = {
                'model': lstm_model,
                'metrics': metrics,
                'sequence_length': sequence_length,
                'model_type': 'LSTM'
            }
            logger.info(f"Trade Impact Model - Train RMSE: {metrics['train_rmse']:.4f}, Val RMSE: {metrics['val_rmse']:.4f}")
            return model_info
        else:
            logger.warning("Insufficient data for LSTM, using XGBoost fallback")
            return self.train_gdp_forecasting_model(X_train, y_train, X_val, y_val)
    def train_agricultural_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        logger.info("Starting model training for all targets...")
        df = self.load_feature_data()
        (X_train, X_val, X_test), (y_train, y_val, y_test), feature_columns = self.prepare_modeling_data(df)
        targets = {
            'GDP_Growth_Percent': 'gdp_forecasting',
            'Composite_Resilience_Score': 'resilience',
            'Trade_Dependency_Index': 'trade_impact',
            'Agricultural_Resilience_Index': 'agricultural'
        }
        for target, model_type in targets.items():
            if target in y_train:
                logger.info(f"Training {model_type} model for {target}...")
                if model_type == 'gdp_forecasting':
                    model_info = self.train_gdp_forecasting_model(X_train, y_train[target], X_val, y_val[target])
                elif model_type == 'resilience':
                    model_info = self.train_resilience_model(X_train, y_train[target], X_val, y_val[target])
                elif model_type == 'trade_impact':
                    model_info = self.train_trade_impact_model(X_train, y_train[target], X_val, y_val[target])
                elif model_type == 'agricultural':
                    model_info = self.train_agricultural_model(X_train, y_train[target], X_val, y_val[target])
                self.models[target] = model_info
                self.feature_importance[target] = model_info.get('feature_importance', None)
        self.save_models()
        return self.models
    def forecast_2030_scenarios(self, scenario: str = 'baseline') -> pd.DataFrame:
        logger.info("Simulating drought impact on agricultural exports...")
        if countries is None:
            countries = ['India', 'USA', 'China', 'Brazil', 'Argentina', 'Australia', 
                       'Canada', 'France', 'Germany', 'Thailand']
        baseline_forecast = self.forecast_2030_scenarios('baseline')
        drought_forecast = self.forecast_2030_scenarios('consecutive_drought')
        baseline = baseline_forecast[baseline_forecast['Country'].isin(countries)]
        drought = drought_forecast[drought_forecast['Country'].isin(countries)]
        impact_analysis = baseline[['Country']].copy()
        impact_analysis['Baseline_Agricultural_Resilience'] = baseline['Agricultural_Resilience_Index_2030_Forecast']
        impact_analysis['Drought_Agricultural_Resilience'] = drought['Agricultural_Resilience_Index_2030_Forecast']
        impact_analysis['Resilience_Impact'] = (
            drought['Agricultural_Resilience_Index_2030_Forecast'] - 
            baseline['Agricultural_Resilience_Index_2030_Forecast']
        )
        impact_analysis['Impact_Percentage'] = (
            impact_analysis['Resilience_Impact'] / 
            (baseline['Agricultural_Resilience_Index_2030_Forecast'] + 1e-8) * 100
        )
        impact_analysis['Estimated_Export_Reduction_Percent'] = (
            -impact_analysis['Impact_Percentage'] * 0.8
        )
        return impact_analysis
    def save_models(self):
        logger.info("Loading trained models...")
        model_files = list(self.models_dir.glob("*_model.pkl"))
        for model_file in model_files:
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                metadata_file = model_file.parent / f"{model_file.stem.replace('_model', '_metadata')}.pkl"
                if metadata_file.exists():
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                else:
                    metadata = {'model_type': 'Unknown', 'target_variable': model_file.stem}
                target = metadata['target_variable']
                self.models[target] = {
                    'model': model,
                    'model_type': metadata['model_type'],
                    'metrics': metadata.get('metrics', {}),
                    'feature_importance': metadata.get('feature_importance', None)
                }
            except Exception as e:
                logger.error(f"Error loading model {model_file}: {e}")
        logger.info(f"Loaded {len(self.models)} models")
    def get_model_performance(self) -> Dict:
    modeler = EconomicModeler()
    models = modeler.train_all_models()
    scenarios = ['baseline', 'increased_social_spending', 'trade_diversification', 'global_crisis']
    all_forecasts = []
    for scenario in scenarios:
        forecast = modeler.forecast_2030_scenarios(scenario)
        all_forecasts.append(forecast)
    drought_impact = modeler.simulate_drought_impact()
    print("\nModel Training Complete!")
    print(f"Trained {len(models)} models")
    print("\nDrought Impact Analysis (Top 10 Countries):")
    print(drought_impact.head(10))
    return models, drought_impact
if __name__ == "__main__":
    main()
