import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class FeatureEngineer:
        try:
            data_path = self.data_dir / "integrated_economic_data.csv"
            if data_path.exists():
                df = pd.read_csv(data_path)
                logger.info(f"Loaded integrated data: {df.shape}")
                return df
            else:
                logger.info("Integrated data not found, running data integration...")
                from data_integration import DataIntegrator
                integrator = DataIntegrator()
                return integrator.integrate_all_data()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    def create_trade_dependency_index(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating resilience score...")
        economic_indicators = [
            'GDP_Growth_Percent', 'GDP_Per_Capita_Growth', 'Trade_Efficiency',
            'Government_Spending_Efficiency', 'Economic_Diversification_Index'
        ]
        social_indicators = [
            'Life_Expectancy_Years', 'HDI_Index', 'Social_Cohesion_Index',
            'Urbanization_Percent', 'Unemployment_Percent'
        ]
        environmental_indicators = [
            'Environmental_Sustainability_Index', 'Disaster_Recovery_Score',
            'Disaster_Damage_Percent_GDP'
        ]
        institutional_indicators = [
            'Institutional_Strength_Index', 'Infrastructure_Quality_Index',
            'Financial_Stability_Index'
        ]
        all_indicators = (economic_indicators + social_indicators + 
                         environmental_indicators + institutional_indicators)
        for col in all_indicators:
            if col in df.columns:
                if col in ['Unemployment_Percent', 'Disaster_Damage_Percent_GDP', 'Gini_Index']:
                    df[f'{col}_Normalized'] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
                else:
                    df[f'{col}_Normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
        df['Economic_Resilience_Score'] = df[[f'{col}_Normalized' for col in economic_indicators if col in df.columns]].mean(axis=1)
        df['Social_Resilience_Score'] = df[[f'{col}_Normalized' for col in social_indicators if col in df.columns]].mean(axis=1)
        df['Environmental_Resilience_Score'] = df[[f'{col}_Normalized' for col in environmental_indicators if col in df.columns]].mean(axis=1)
        df['Institutional_Resilience_Score'] = df[[f'{col}_Normalized' for col in institutional_indicators if col in df.columns]].mean(axis=1)
        df['Composite_Resilience_Score'] = (
            df['Economic_Resilience_Score'] * 0.35 +
            df['Social_Resilience_Score'] * 0.25 +
            df['Environmental_Resilience_Score'] * 0.20 +
            df['Institutional_Resilience_Score'] * 0.20
        )
        return df
    def create_spending_efficiency_index(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating shock impact score...")
        crisis_years = [2008, 2009, 2020, 2021]
        df['Crisis_Impact_Score'] = 0.0
        for country in df['Country'].unique():
            country_data = df[df['Country'] == country]
            for crisis_year in crisis_years:
                if crisis_year in country_data['Year'].values:
                    pre_crisis = country_data[country_data['Year'] == crisis_year - 1]
                    crisis = country_data[country_data['Year'] == crisis_year]
                    if not pre_crisis.empty and not crisis.empty:
                        gdp_impact = (crisis['GDP_Growth_Percent'].iloc[0] - pre_crisis['GDP_Growth_Percent'].iloc[0]) / (abs(pre_crisis['GDP_Growth_Percent'].iloc[0]) + 1e-8)
                        trade_impact = (crisis['Trade_Percent_GDP'].iloc[0] - pre_crisis['Trade_Percent_GDP'].iloc[0]) / (abs(pre_crisis['Trade_Percent_GDP'].iloc[0]) + 1e-8)
                        crisis_idx = crisis.index[0]
                        df.loc[crisis_idx, 'Crisis_Impact_Score'] = abs(gdp_impact) + abs(trade_impact)
        df['Recovery_Speed_Score'] = 1 / (df['Recovery_Years'] + 1)
        df['Disaster_Resilience_Score'] = 1 / (df['Disaster_Damage_Percent_GDP'] + 1)
        shock_metrics = ['Crisis_Impact_Score', 'Recovery_Speed_Score', 'Disaster_Resilience_Score']
        for col in shock_metrics:
            if col in df.columns:
                df[f'{col}_Normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
        df['Shock_Impact_Score'] = (
            df['Crisis_Impact_Score_Normalized'] * 0.4 +
            (1 - df['Recovery_Speed_Score_Normalized']) * 0.3 +
            (1 - df['Disaster_Resilience_Score_Normalized']) * 0.3
        )
        return df
    def create_agricultural_resilience_index(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating time series features...")
        lag_columns = [
            'GDP_Growth_Percent', 'Inflation_Percent', 'Unemployment_Percent',
            'Exports_Billions_USD', 'Imports_Billions_USD', 'Trade_Balance_Billions_USD'
        ]
        for col in lag_columns:
            if col in df.columns:
                df[f'{col}_Lag1'] = df.groupby('Country')[col].shift(1)
                df[f'{col}_Lag2'] = df.groupby('Country')[col].shift(2)
                df[f'{col}_Lag3'] = df.groupby('Country')[col].shift(3)
        for col in lag_columns:
            if col in df.columns:
                df[f'{col}_Rolling_3Y'] = df.groupby('Country')[col].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
                df[f'{col}_Rolling_5Y'] = df.groupby('Country')[col].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
        for col in lag_columns:
            if col in df.columns:
                df[f'{col}_Trend_5Y'] = df.groupby('Country')[col].rolling(window=5, min_periods=3).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                ).reset_index(0, drop=True)
        return df
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating cluster features...")
        cluster_features = [
            'GDP_Per_Capita_USD', 'Trade_Percent_GDP', 'Urbanization_Percent',
            'Life_Expectancy_Years', 'HDI_Index'
        ]
        available_features = [col for col in cluster_features if col in df.columns]
        if len(available_features) >= 3:
            cluster_data = df[available_features].fillna(df[available_features].median())
            cluster_data_scaled = self.scaler.fit_transform(cluster_data)
            n_clusters = 5
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['Economic_Cluster'] = kmeans.fit_predict(cluster_data_scaled)
            cluster_dummies = pd.get_dummies(df['Economic_Cluster'], prefix='Cluster')
            df = pd.concat([df, cluster_dummies], axis=1)
            df['Cluster_Size'] = df.groupby('Economic_Cluster')['Economic_Cluster'].transform('count')
            df['Cluster_Average_GDP'] = df.groupby('Economic_Cluster')['GDP_Per_Capita_USD'].transform('mean')
        return df
    def engineer_all_features(self) -> pd.DataFrame:
        if self.feature_data is not None:
            output_path = self.data_dir / filename
            self.feature_data.to_csv(output_path, index=False)
            logger.info(f"Feature data saved to: {output_path}")
        else:
            logger.warning("No feature data to save. Run engineer_all_features() first.")
    def get_feature_summary(self) -> Dict:
    engineer = FeatureEngineer()
    feature_data = engineer.engineer_all_features()
    engineer.save_feature_data()
    summary = engineer.get_feature_summary()
    print("\nFeature Engineering Summary:")
    print(f"Total features: {summary['total_features']}")
    print(f"Total records: {summary['total_records']}")
    print("\nFeature categories:")
    for category, count in summary['feature_counts'].items():
        print(f"  {category}: {count} features")
    return feature_data
if __name__ == "__main__":
    main()
