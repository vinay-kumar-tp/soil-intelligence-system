import os
import sys
from pathlib import Path
from environment import settings, PROJECT_ROOT
from diagnostics import get_system_diagnostics

def run_startup_validation():
    """Validate the operational environment before starting services."""
    issues = []
    
    # 1. Dependency Verification
    try:
        import xgboost
        import fastapi
        import pandas
        import uvicorn
        import streamlit
    except ImportError as e:
        issues.append(f"[CRITICAL] Missing core dependency: {e}")
        
    # 2. Virtual Environment Verification
    diag = get_system_diagnostics()
    if not diag["is_venv"] and settings.ENVIRONMENT != "container":
        issues.append("[WARNING] Application is not running inside a virtual environment.")
        
    # 3. Path & Registry Verification
    required_dirs = [
        settings.MODEL_REGISTRY_PATH,
        settings.ARTIFACT_REGISTRY_PATH
    ]
    
    for d in required_dirs:
        if not d.exists():
            issues.append(f"[CRITICAL] Required registry directory missing: {d}")
            
    # 4. Model Verification (Basic check)
    crop_model = settings.MODEL_REGISTRY_PATH / "crop_pipeline" / "xgboost_v1.pkl"
    if not crop_model.exists():
        issues.append("[WARNING] Primary crop model (xgboost_v1.pkl) not found. System may degrade.")
        
    return issues

def generate_startup_report(issues, filepath):
    """Generate a structured text report of the startup validation."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    diag = get_system_diagnostics()
    
    with open(filepath, "w") as f:
        f.write("=== STARTUP VALIDATION REPORT ===\n")
        f.write(f"Environment: {settings.ENVIRONMENT}\n")
        f.write(f"Python: {diag['python_version']} ({diag['python_executable']})\n")
        f.write(f"Venv Active: {diag['is_venv']}\n\n")
        
        f.write("--- ISSUES DETECTED ---\n")
        if not issues:
            f.write("No issues detected. System is operational.\n")
        else:
            for issue in issues:
                f.write(f"{issue}\n")
                
if __name__ == "__main__":
    issues = run_startup_validation()
    report_path = PROJECT_ROOT / "reports" / "startup_validation_report.txt"
    generate_startup_report(issues, report_path)
    if issues and any("[CRITICAL]" in i for i in issues):
        print(f"Startup validation FAILED. See {report_path}")
        sys.exit(1)
    else:
        print(f"Startup validation PASSED. See {report_path}")
        sys.exit(0)
