import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "source_code"))
def test_imports():
    print("\nTesting class initialization...")
    try:
        from data_integration import DataIntegrator
        integrator = DataIntegrator()
        print("✅ DataIntegrator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize DataIntegrator: {e}")
        return False
    try:
        from feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        print("✅ FeatureEngineer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize FeatureEngineer: {e}")
        return False
    try:
        from modeling import EconomicModeler
        modeler = EconomicModeler()
        print("✅ EconomicModeler initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize EconomicModeler: {e}")
        return False
    try:
        from visualization import EconomicVisualizer
        visualizer = EconomicVisualizer()
        print("✅ EconomicVisualizer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize EconomicVisualizer: {e}")
        return False
    try:
        from policy_recommendations import PolicyAdvisor
        advisor = PolicyAdvisor()
        print("✅ PolicyAdvisor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize PolicyAdvisor: {e}")
        return False
    return True
def test_directory_structure():
    print("\nTesting source files...")
    required_files = [
        "source_code/data_integration.py",
        "source_code/feature_engineering.py", 
        "source_code/modeling.py",
        "source_code/visualization.py",
        "source_code/policy_recommendations.py",
        "source_code/main.py",
        "requirements.txt",
        "README.md"
    ]
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ Missing file: {file_path}")
            return False
    return True
def main():
