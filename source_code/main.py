import sys
import logging
from pathlib import Path
from datetime import datetime
import time
sys.path.append(str(Path(__file__).parent))
from data_integration import DataIntegrator
from feature_engineering import FeatureEngineer
from modeling import EconomicModeler
from visualization import EconomicVisualizer
from policy_recommendations import PolicyAdvisor
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('economic_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
class EconomicResiliencePipeline:
        logger.info("Starting complete economic resilience analysis...")
        try:
            logger.info("=" * 60)
            logger.info("STEP 1: DATA INTEGRATION")
            logger.info("=" * 60)
            integrated_data = self.data_integrator.integrate_all_data()
            logger.info(f"Data integration complete. Dataset shape: {integrated_data.shape}")
            logger.info("=" * 60)
            logger.info("STEP 2: FEATURE ENGINEERING")
            logger.info("=" * 60)
            feature_data = self.feature_engineer.engineer_all_features()
            logger.info(f"Feature engineering complete. Dataset shape: {feature_data.shape}")
            logger.info("=" * 60)
            logger.info("STEP 3: MODEL TRAINING")
            logger.info("=" * 60)
            models = self.modeler.train_all_models()
            logger.info(f"Model training complete. Trained {len(models)} models")
            logger.info("=" * 60)
            logger.info("STEP 4: FORECAST GENERATION")
            logger.info("=" * 60)
            scenarios = ['baseline', 'increased_social_spending', 'trade_diversification', 'global_crisis']
            all_forecasts = []
            for scenario in scenarios:
                forecast = self.modeler.forecast_2030_scenarios(scenario)
                all_forecasts.append(forecast)
                logger.info(f"Generated {scenario} forecast for {len(forecast)} countries")
            combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
            combined_forecasts.to_csv(self.data_dir / "2030_forecasts.csv", index=False)
            logger.info("All forecasts saved to 2030_forecasts.csv")
            logger.info("=" * 60)
            logger.info("STEP 5: DROUGHT IMPACT ANALYSIS")
            logger.info("=" * 60)
            drought_impact = self.modeler.simulate_drought_impact()
            logger.info(f"Drought impact analysis complete for {len(drought_impact)} countries")
            logger.info("=" * 60)
            logger.info("STEP 6: VISUALIZATION")
            logger.info("=" * 60)
            charts = self.visualizer.create_all_visualizations()
            logger.info(f"Visualization complete. Created {len(charts)} charts")
            logger.info("=" * 60)
            logger.info("STEP 7: POLICY RECOMMENDATIONS")
            logger.info("=" * 60)
            recommendations = self.policy_advisor.create_all_recommendations()
            logger.info("Policy recommendations complete")
            logger.info("=" * 60)
            logger.info("STEP 8: SUMMARY REPORT")
            logger.info("=" * 60)
            self._generate_summary_report(integrated_data, feature_data, models, 
                                       combined_forecasts, drought_impact, recommendations)
            execution_time = time.time() - self.start_time
            logger.info(f"Complete analysis finished in {execution_time:.2f} seconds")
            return {
                'integrated_data': integrated_data,
                'feature_data': feature_data,
                'models': models,
                'forecasts': combined_forecasts,
                'drought_impact': drought_impact,
                'recommendations': recommendations,
                'execution_time': execution_time
            }
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}")
            raise
    def run_data_integration_only(self):
        logger.info("Running feature engineering only...")
        return self.feature_engineer.engineer_all_features()
    def run_modeling_only(self):
        logger.info("Running visualization only...")
        return self.visualizer.create_all_visualizations()
    def run_policy_recommendations_only(self):
        logger.info("Generating summary report...")
        report_content = f"""
ECONOMIC RESILIENCE ANALYSIS - SUMMARY REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Execution time: {time.time() - self.start_time:.2f} seconds
1. DATA OVERVIEW
===============
- Integrated dataset: {integrated_data.shape[0]} records, {integrated_data.shape[1]} columns
- Feature-engineered dataset: {feature_data.shape[0]} records, {feature_data.shape[1]} columns
- Countries analyzed: {len(integrated_data['Country'].unique())}
- Time period: {integrated_data['Year'].min()} - {integrated_data['Year'].max()}
2. MODEL PERFORMANCE
===================
- Models trained: {len(models)}
- Model types: {', '.join([info['model_type'] for info in models.values()])}
3. FORECAST RESULTS
==================
- Scenarios analyzed: {len(forecasts['Scenario'].unique())}
- Countries forecasted: {len(forecasts['Country'].unique())}
- Forecast horizon: 2030
4. DROUGHT IMPACT ANALYSIS
==========================
- Countries analyzed: {len(drought_impact)}
- Average resilience impact: {drought_impact['Resilience_Impact'].mean():.3f}
- Average export reduction: {drought_impact['Estimated_Export_Reduction_Percent'].mean():.1f}%
5. POLICY RECOMMENDATIONS
=========================
- Countries with recommendations: {len(recommendations.get('country_analyses', {}))}
- High-risk countries: {sum(1 for analysis in recommendations.get('country_analyses', {}).values() 
                           if analysis.get('overall_risk_level') in ['High', 'Critical'])}
6. OUTPUT FILES
===============
- Data files: {self.data_dir}
- Visualizations: {self.outputs_dir}/visualizations
- Reports: {self.outputs_dir}/reports
- Models: models/
7. KEY INSIGHTS
===============
- Economic vulnerabilities are the most common across countries
- Climate resilience and institutional strengthening are priority areas
- Cross-country coordination is essential for global resilience
- Immediate interventions needed for high-risk countries
8. NEXT STEPS
=============
- Review policy recommendations for each country
- Implement high-priority interventions
- Monitor progress and adjust strategies
- Strengthen international coordination mechanisms
---
Report generated by Economic Resilience Pipeline
Trade Resilience & Economic Networks Team
    print("Economic Resilience Analysis Pipeline")
    print("Trade Resilience & Economic Networks Team")
    print("=" * 60)
    pipeline = EconomicResiliencePipeline()
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == 'data_only':
            pipeline.run_data_integration_only()
        elif mode == 'features_only':
            pipeline.run_feature_engineering_only()
        elif mode == 'modeling_only':
            pipeline.run_modeling_only()
        elif mode == 'visualization_only':
            pipeline.run_visualization_only()
        elif mode == 'policy_only':
            pipeline.run_policy_recommendations_only()
        elif mode == 'help':
            print("""
Usage: python main.py [mode]
Modes:
- data_only: Run only data integration
- features_only: Run only feature engineering
- modeling_only: Run only modeling
- visualization_only: Run only visualization
- policy_only: Run only policy recommendations
- help: Show this help message
- (no argument): Run complete pipeline
Examples:
  python main.py
  python main.py data_only
  python main.py modeling_only
