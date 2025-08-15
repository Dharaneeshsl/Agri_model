import subprocess
import sys
import os
from pathlib import Path
def install_package(package):
    print("Installing required packages...")
    core_packages = [
        "pandas",
        "numpy", 
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ]
    print("\nInstalling core packages...")
    for package in core_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
    advanced_packages = [
        "xgboost",
        "lightgbm", 
        "tensorflow",
        "statsmodels",
        "plotly",
        "folium"
    ]
    print("\nInstalling advanced ML packages...")
    for package in advanced_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
        else:
            print(f"⚠️  Failed to install {package} (optional)")
    print("\nDependency installation complete!")
    print("\nNote: Some packages may require additional system dependencies.")
    print("If you encounter issues, try installing them manually:")
def create_minimal_requirements():
    with open("requirements_minimal.txt", "w") as f:
        f.write(minimal_requirements)
    print("Created requirements_minimal.txt with basic dependencies")
def check_python_version():
    print("=" * 60)
    print("ECONOMIC RESILIENCE PIPELINE - DEPENDENCY INSTALLER")
    print("=" * 60)
    if not check_python_version():
        return 1
    print("\nThis script will install the required packages for the Economic Resilience Pipeline.")
    print("You can also install them manually using:")
    print("  pip install -r requirements.txt")
    create_minimal_requirements()
    response = input("\nDo you want to install dependencies now? (y/n): ").lower()
    if response in ['y', 'yes']:
        install_requirements()
        print("\n🎉 Dependencies installed! You can now run the pipeline.")
        print("\nTo test the installation:")
        print("  python test_pipeline.py")
        print("\nTo run the complete analysis:")
        print("  python source_code/main.py")
    else:
        print("\nDependencies not installed.")
        print("Install them manually using:")
        print("  pip install -r requirements.txt")
        print("
        print("  pip install -r requirements_minimal.txt")
    return 0
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
