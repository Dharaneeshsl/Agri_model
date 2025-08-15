import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import folium
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
class EconomicVisualizer:
        try:
            feature_path = self.data_dir / "engineered_features.csv"
            if feature_path.exists():
                self.feature_data = pd.read_csv(feature_path)
                logger.info(f"Loaded feature data: {self.feature_data.shape}")
            else:
                logger.warning("Feature data not found")
            forecast_path = self.data_dir / "2030_forecasts.csv"
            if forecast_path.exists():
                self.forecast_data = pd.read_csv(forecast_path)
                logger.info(f"Loaded forecast data: {self.forecast_data.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    def create_resilience_heatmap(self, save: bool = True) -> go.Figure:
        logger.info("Creating trade network graph...")
        if self.feature_data is None:
            logger.error("No feature data available")
            return None
        latest_data = self.feature_data[self.feature_data['Year'] == 2024].copy()
        if latest_data.empty:
            latest_data = self.feature_data[self.feature_data['Year'] == self.feature_data['Year'].max()].copy()
        countries = latest_data['Country'].unique()
        trade_data = []
        for i, country1 in enumerate(countries):
            for j, country2 in enumerate(countries):
                if i != j:
                    country1_data = latest_data[latest_data['Country'] == country1].iloc[0]
                    country2_data = latest_data[latest_data['Country'] == country2].iloc[0]
                    trade_volume = (
                        country1_data.get('Exports_Billions_USD', 0) * 
                        country2_data.get('Imports_Billions_USD', 0) / 1000
                    )
                    if trade_volume > 0.1:
                        trade_data.append({
                            'source': country1,
                            'target': country2,
                            'value': trade_volume
                        })
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=countries,
                color="blue"
            ),
            link=dict(
                source=[countries.tolist().index(link['source']) for link in trade_data],
                target=[countries.tolist().index(link['target']) for link in trade_data],
                value=[link['value'] for link in trade_data]
            )
        )])
        fig.update_layout(
            title_text="Global Trade Network (2024)",
            font_size=10,
            height=800
        )
        if save:
            fig.write_html(self.visualizations_dir / "trade_network.html")
        return fig
    def create_gdp_growth_timeline(self, save: bool = True) -> go.Figure:
        logger.info("Creating vulnerability radar chart...")
        if self.feature_data is None:
            logger.error("No feature data available")
            return None
        if countries is None:
            countries = ['USA', 'China', 'India', 'Germany', 'Japan']
        latest_data = self.feature_data[self.feature_data['Year'] == 2024].copy()
        if latest_data.empty:
            latest_data = self.feature_data[self.feature_data['Year'] == self.feature_data['Year'].max()].copy()
        country_data = latest_data[latest_data['Country'].isin(countries)]
        vulnerability_dimensions = [
            'Trade_Dependency_Index', 'Crisis_Vulnerability_Score',
            'Disaster_Damage_Percent_GDP', 'Unemployment_Percent',
            'Inflation_Percent'
        ]
        available_dimensions = [col for col in vulnerability_dimensions if col in country_data.columns]
        if not available_dimensions:
            logger.warning("No vulnerability dimensions found")
            return None
        fig = go.Figure()
        for _, country_row in country_data.iterrows():
            values = [country_row[dim] for dim in available_dimensions]
            values.append(values[0])
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=available_dimensions + [available_dimensions[0]],
                fill='toself',
                name=country_row['Country']
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Vulnerability Radar Chart by Country (2024)",
            height=600
        )
        if save:
            fig.write_html(self.visualizations_dir / "vulnerability_radar.html")
            fig.write_image(self.visualizations_dir / "vulnerability_radar.png", width=800, height=600)
        return fig
    def create_scenario_comparison(self, save: bool = True) -> go.Figure:
        logger.info("Creating drought impact analysis...")
        countries = ['India', 'USA', 'China', 'Brazil', 'Argentina', 'Australia', 
                   'Canada', 'France', 'Germany', 'Thailand']
        np.random.seed(42)
        drought_data = pd.DataFrame({
            'Country': countries,
            'Baseline_Resilience': np.random.uniform(0.6, 0.9, len(countries)),
            'Drought_Resilience': np.random.uniform(0.3, 0.7, len(countries)),
            'Export_Reduction': np.random.uniform(15, 45, len(countries))
        })
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Resilience Impact', 'Export Reduction'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        fig.add_trace(
            go.Bar(
                x=drought_data['Country'],
                y=drought_data['Baseline_Resilience'],
                name='Baseline',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(
                x=drought_data['Country'],
                y=drought_data['Drought_Resilience'],
                name='Drought Impact',
                marker_color='red'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(
                x=drought_data['Country'],
                y=drought_data['Export_Reduction'],
                name='Export Reduction (%)',
                marker_color='orange'
            ),
            row=1, col=2
        )
        fig.update_layout(
            title_text="Drought Impact on Agricultural Exports (2030)",
            title_x=0.5,
            height=500,
            showlegend=True
        )
        if save:
            fig.write_html(self.visualizations_dir / "drought_impact_analysis.html")
            fig.write_image(self.visualizations_dir / "drought_impact_analysis.png", width=1200, height=600)
        return fig
    def create_feature_importance_charts(self, save: bool = True) -> go.Figure:
        logger.info("Creating geographic disaster map...")
        disaster_locations = {
            'USA': (39.8283, -98.5795),
            'China': (35.8617, 104.1954),
            'India': (20.5937, 78.9629),
            'Japan': (36.2048, 138.2529),
            'Germany': (51.1657, 10.4515),
            'Brazil': (-14.2350, -51.9253),
            'Australia': (-25.2744, 133.7751),
            'Canada': (56.1304, -106.3468)
        }
        m = folium.Map(location=[20, 0], zoom_start=2)
        for country, coords in disaster_locations.items():
            impact_level = np.random.choice(['Low', 'Medium', 'High', 'Extreme'])
            impact_colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'orange', 'Extreme': 'red'}
            folium.Marker(
                coords,
                popup=f"{country}<br>Disaster Impact: {impact_level}",
                icon=folium.Icon(color=impact_colors[impact_level], icon='info-sign')
            ).add_to(m)
        if save:
            m.save(str(self.visualizations_dir / "disaster_map.html"))
        return m
    def _create_sample_forecasts(self):
        logger.info("Creating all visualizations...")
        self.load_data()
        charts = [
            self.create_resilience_heatmap(),
            self.create_trade_network_graph(),
            self.create_gdp_growth_timeline(),
            self.create_vulnerability_radar_chart(),
            self.create_scenario_comparison(),
            self.create_drought_impact_analysis(),
            self.create_feature_importance_charts()
        ]
        self.create_geographic_disaster_map()
        logger.info(f"All visualizations created and saved to {self.visualizations_dir}")
        return charts
def main():
