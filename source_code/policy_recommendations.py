import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class PolicyAdvisor:
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
    def analyze_country_vulnerabilities(self, country: str) -> Dict:
        logger.info(f"Generating recommendations for {country}")
        vulnerabilities = self.analyze_country_vulnerabilities(country)
        if not vulnerabilities:
            return {}
        recommendations = {
            'country': country,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'overall_risk_level': self._calculate_overall_risk(vulnerabilities),
            'vulnerabilities': vulnerabilities,
            'policy_recommendations': {},
            'implementation_priority': [],
            'expected_impact': {},
            'timeline': {}
        }
        for category, vuln_info in vulnerabilities.items():
            if vuln_info['priority'] in ['High', 'Medium']:
                recommendations['policy_recommendations'][category] = self._get_category_recommendations(
                    category, vuln_info, country
                )
        recommendations['implementation_priority'] = self._set_implementation_priorities(vulnerabilities)
        recommendations['expected_impact'] = self._estimate_policy_impact(vulnerabilities)
        recommendations['timeline'] = self._set_implementation_timeline(vulnerabilities)
        return recommendations
    def _calculate_overall_risk(self, vulnerabilities: Dict) -> str:
        recommendations = {
            'Economic': [
                {
                    'policy': 'Fiscal Stimulus Package',
                    'description': 'Implement targeted fiscal stimulus to boost GDP growth and reduce unemployment',
                    'budget_estimate': '2-5% of GDP',
                    'implementation_time': '6-12 months',
                    'expected_outcome': 'Increase GDP growth by 0.5-1.5 percentage points'
                },
                {
                    'policy': 'Trade Diversification Strategy',
                    'description': 'Reduce trade dependency by expanding trade partnerships and developing domestic markets',
                    'budget_estimate': '1-3% of GDP',
                    'implementation_time': '12-24 months',
                    'expected_outcome': 'Reduce trade dependency by 15-25%'
                },
                {
                    'policy': 'Inflation Control Measures',
                    'description': 'Implement monetary policy measures to control inflation within target range',
                    'budget_estimate': '0.5-1% of GDP',
                    'implementation_time': '3-6 months',
                    'expected_outcome': 'Reduce inflation to target range of 2-3%'
                }
            ],
            'Social': [
                {
                    'policy': 'Human Development Investment',
                    'description': 'Increase investment in education, healthcare, and social services',
                    'budget_estimate': '3-6% of GDP',
                    'implementation_time': '24-36 months',
                    'expected_outcome': 'Improve HDI by 0.05-0.1 points'
                },
                {
                    'policy': 'Income Inequality Reduction',
                    'description': 'Implement progressive taxation and social welfare programs',
                    'budget_estimate': '2-4% of GDP',
                    'implementation_time': '18-30 months',
                    'expected_outcome': 'Reduce Gini coefficient by 3-5 points'
                }
            ],
            'Environmental': [
                {
                    'policy': 'Climate Resilience Infrastructure',
                    'description': 'Invest in climate-resilient infrastructure and disaster preparedness',
                    'budget_estimate': '2-4% of GDP',
                    'implementation_time': '36-48 months',
                    'expected_outcome': 'Reduce disaster damage by 20-30%'
                },
                {
                    'policy': 'Sustainable Development Programs',
                    'description': 'Promote renewable energy and sustainable resource management',
                    'budget_estimate': '1-3% of GDP',
                    'implementation_time': '24-36 months',
                    'expected_outcome': 'Improve environmental sustainability by 15-25%'
                }
            ],
            'Institutional': [
                {
                    'policy': 'Governance Reform',
                    'description': 'Strengthen institutional frameworks and anti-corruption measures',
                    'budget_estimate': '1-2% of GDP',
                    'implementation_time': '24-36 months',
                    'expected_outcome': 'Improve institutional strength by 10-20%'
                },
                {
                    'policy': 'Infrastructure Modernization',
                    'description': 'Upgrade critical infrastructure including digital and physical assets',
                    'budget_estimate': '3-6% of GDP',
                    'implementation_time': '36-60 months',
                    'expected_outcome': 'Improve infrastructure quality by 20-30%'
                }
            ]
        }
        if vuln_info['priority'] == 'High':
            return recommendations.get(category, [])
        else:
            return recommendations.get(category, [])[:2]
    def _set_implementation_priorities(self, vulnerabilities: Dict) -> List[str]:
        impact_estimates = {}
        for category, vuln_info in vulnerabilities.items():
            if vuln_info['priority'] in ['High', 'Medium']:
                base_improvement = 0.15 if vuln_info['priority'] == 'High' else 0.10
                impact_estimates[category] = {
                    'vulnerability_reduction': f"{base_improvement * 100:.1f}%",
                    'resilience_improvement': f"{base_improvement * 100:.1f}%",
                    'economic_benefit': f"${base_improvement * 1000:.0f}M - ${base_improvement * 5000:.0f}M",
                    'time_to_impact': '12-24 months'
                }
        return impact_estimates
    def _set_implementation_timeline(self, vulnerabilities: Dict) -> Dict:
        logger.info(f"Creating crisis scenario analysis for {scenario}")
        scenarios = {
            'global_recession': {
                'description': 'Global economic downturn with 3-5% GDP contraction',
                'duration': '18-24 months',
                'key_impacts': ['Reduced trade volumes', 'Increased unemployment', 'Lower commodity prices'],
                'policy_responses': [
                    'Expansionary fiscal policy',
                    'Monetary easing',
                    'Trade protection measures',
                    'Social safety net expansion'
                ]
            },
            'climate_disaster': {
                'description': 'Severe climate events affecting multiple regions',
                'duration': '12-36 months',
                'key_impacts': ['Agricultural production loss', 'Infrastructure damage', 'Population displacement'],
                'policy_responses': [
                    'Disaster relief funding',
                    'Climate adaptation infrastructure',
                    'Agricultural diversification',
                    'Insurance and risk pooling'
                ]
            },
            'trade_war': {
                'description': 'Escalating trade tensions and tariffs',
                'duration': '24-48 months',
                'key_impacts': ['Supply chain disruption', 'Increased costs', 'Market uncertainty'],
                'policy_responses': [
                    'Trade diversification',
                    'Domestic production support',
                    'Alternative supply chains',
                    'Diplomatic engagement'
                ]
            },
            'pandemic_outbreak': {
                'description': 'Global health crisis with economic implications',
                'duration': '12-24 months',
                'key_impacts': ['Healthcare system strain', 'Economic activity reduction', 'Social distancing measures'],
                'policy_responses': [
                    'Healthcare system strengthening',
                    'Digital transformation support',
                    'Remote work infrastructure',
                    'Economic stimulus packages'
                ]
            }
        }
        if scenario not in scenarios:
            logger.warning(f"Unknown scenario: {scenario}")
            return {}
        scenario_info = scenarios[scenario]
        policy_effectiveness = {}
        for policy in scenario_info['policy_responses']:
            effectiveness_score = np.random.uniform(0.6, 0.9)
            implementation_cost = np.random.uniform(0.5, 2.0)
            time_to_effect = np.random.choice(['3-6 months', '6-12 months', '12-24 months'])
            policy_effectiveness[policy] = {
                'effectiveness_score': effectiveness_score,
                'implementation_cost': f"{implementation_cost:.1f}% of GDP",
                'time_to_effect': time_to_effect,
                'risk_level': 'Low' if effectiveness_score > 0.8 else 'Medium' if effectiveness_score > 0.7 else 'High'
            }
        return {
            'scenario': scenario,
            'scenario_info': scenario_info,
            'policy_effectiveness': policy_effectiveness,
            'recommended_actions': self._get_scenario_recommendations(scenario, policy_effectiveness)
        }
    def _get_scenario_recommendations(self, scenario: str, policy_effectiveness: Dict) -> List[str]:
        logger.info("Generating comprehensive policy recommendations report")
        if countries is None:
            countries = ['USA', 'China', 'India', 'Germany', 'Japan', 'UK', 'France', 'Italy']
        comprehensive_report = {
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'executive_summary': {},
            'country_analyses': {},
            'cross_country_recommendations': {},
            'crisis_scenarios': {},
            'implementation_roadmap': {}
        }
        for country in countries:
            country_rec = self.generate_country_recommendations(country)
            if country_rec:
                comprehensive_report['country_analyses'][country] = country_rec
        comprehensive_report['cross_country_recommendations'] = self._generate_cross_country_recommendations()
        crisis_scenarios = ['global_recession', 'climate_disaster', 'trade_war', 'pandemic_outbreak']
        for scenario in crisis_scenarios:
            comprehensive_report['crisis_scenarios'][scenario] = self.create_crisis_scenario_analysis(scenario)
        comprehensive_report['implementation_roadmap'] = self._create_implementation_roadmap()
        comprehensive_report['executive_summary'] = self._generate_executive_summary(comprehensive_report)
        return comprehensive_report
    def _generate_cross_country_recommendations(self) -> Dict:
        return {
            'phase_1_immediate': {
                'timeline': '0-6 months',
                'actions': [
                    'Establish policy coordination committees',
                    'Conduct vulnerability assessments',
                    'Allocate emergency funding',
                    'Begin high-priority interventions'
                ]
            },
            'phase_2_short_term': {
                'timeline': '6-18 months',
                'actions': [
                    'Implement core policy reforms',
                    'Strengthen institutional capacity',
                    'Begin infrastructure projects',
                    'Monitor and evaluate progress'
                ]
            },
            'phase_3_medium_term': {
                'timeline': '18-36 months',
                'actions': [
                    'Complete major infrastructure projects',
                    'Scale up successful interventions',
                    'Strengthen international partnerships',
                    'Establish long-term monitoring systems'
                ]
            },
            'phase_4_long_term': {
                'timeline': '36-60 months',
                'actions': [
                    'Achieve resilience targets',
                    'Establish sustainable systems',
                    'Share best practices globally',
                    'Plan for future challenges'
                ]
            }
        }
    def _generate_executive_summary(self, report: Dict) -> Dict:
        if filename is None:
            filename = f"policy_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = self.reports_dir / filename
        with open(output_path, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        logger.info(f"Recommendations saved to: {output_path}")
    def create_all_recommendations(self):
    advisor = PolicyAdvisor()
    recommendations = advisor.create_all_recommendations()
    print("\nPolicy Recommendations Complete!")
    print(f"Reports saved to: {advisor.reports_dir}")
    if 'executive_summary' in recommendations:
        summary = recommendations['executive_summary']
        print(f"\nExecutive Summary:")
        print(f"Countries analyzed: {summary.get('total_countries_analyzed', 0)}")
        print(f"High-risk countries: {summary.get('high_risk_countries', 0)}")
        print("\nKey findings:")
        for finding in summary.get('key_findings', []):
            print(f"  • {finding}")
    return recommendations
if __name__ == "__main__":
    main()
