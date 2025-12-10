"""
Verification Script for Hull Tactical Market Prediction Codebase
This script tests all critical imports and verifies the environment is ready.
"""

import sys
from pathlib import Path

print("=" * 70)
print("HULL TACTICAL MARKET PREDICTION - ENVIRONMENT VERIFICATION")
print("=" * 70)
print()

# Test 1: Core Data Science Libraries
print("📦 Testing Core Libraries...")
try:
    import pandas as pd
    print(f"  ✅ pandas {pd.__version__}")
except ImportError as e:
    print(f"  ❌ pandas: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"  ✅ numpy {np.__version__}")
except ImportError as e:
    print(f"  ❌ numpy: {e}")
    sys.exit(1)

try:
    import polars as pl
    print(f"  ✅ polars {pl.__version__}")
except ImportError as e:
    print(f"  ❌ polars: {e}")
    sys.exit(1)

print()

# Test 2: Machine Learning Libraries
print("🤖 Testing ML Libraries...")
try:
    import sklearn
    print(f"  ✅ scikit-learn {sklearn.__version__}")
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import TimeSeriesSplit
    print(f"  ✅ All sklearn imports working")
except ImportError as e:
    print(f"  ❌ sklearn: {e}")
    sys.exit(1)

try:
    import lightgbm as lgb
    print(f"  ✅ lightgbm {lgb.__version__}")
except ImportError as e:
    print(f"  ❌ lightgbm: {e}")
    sys.exit(1)

print()

# Test 3: Visualization Libraries
print("📊 Testing Visualization Libraries...")
try:
    import matplotlib
    import matplotlib.pyplot as plt
    print(f"  ✅ matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"  ❌ matplotlib: {e}")
    sys.exit(1)

try:
    import seaborn as sns
    print(f"  ✅ seaborn {sns.__version__}")
except ImportError as e:
    print(f"  ❌ seaborn: {e}")
    sys.exit(1)

print()

# Test 4: Utility Libraries
print("🔧 Testing Utility Libraries...")
try:
    from tqdm import tqdm
    print(f"  ✅ tqdm")
except ImportError as e:
    print(f"  ❌ tqdm: {e}")
    sys.exit(1)

print()

# Test 5: Check Data Files
print("📁 Checking Data Files...")
data_dir = Path("./hull-tactical-market-prediction")
train_file = data_dir / "train.csv"
test_file = data_dir / "test.csv"

if train_file.exists():
    print(f"  ✅ train.csv found ({train_file.stat().st_size / 1024 / 1024:.2f} MB)")
else:
    print(f"  ⚠️  train.csv not found at {train_file}")

if test_file.exists():
    print(f"  ✅ test.csv found ({test_file.stat().st_size / 1024:.2f} KB)")
else:
    print(f"  ⚠️  test.csv not found at {test_file}")

print()

# Test 6: Check Kaggle Evaluation Module
print("🔌 Checking Kaggle Evaluation Module...")
kaggle_eval_dir = data_dir / "kaggle_evaluation"
if kaggle_eval_dir.exists():
    print(f"  ✅ kaggle_evaluation directory found")
    sys.path.insert(0, str(data_dir))
    try:
        import kaggle_evaluation.default_inference_server
        print(f"  ✅ kaggle_evaluation module imports successfully")
    except ImportError as e:
        print(f"  ⚠️  kaggle_evaluation import issue (expected locally): {e}")
else:
    print(f"  ⚠️  kaggle_evaluation directory not found at {kaggle_eval_dir}")

print()

# Test 7: Quick Functionality Test
print("⚡ Running Quick Functionality Test...")
try:
    # Create sample data
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randn(100)
    })
    
    # Test ElasticNet
    X = df[['feature1', 'feature2']]
    y = df['target']
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(X, y)
    pred = model.predict(X[:5])
    print(f"  ✅ ElasticNet model trained and predicted successfully")
    
    # Test LightGBM
    lgb_model = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
    lgb_model.fit(X, y)
    lgb_pred = lgb_model.predict(X[:5])
    print(f"  ✅ LightGBM model trained and predicted successfully")
    
    # Test TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X))
    print(f"  ✅ TimeSeriesSplit working ({len(splits)} splits)")
    
except Exception as e:
    print(f"  ❌ Functionality test failed: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("✅ ALL TESTS PASSED! Your environment is ready!")
print("=" * 70)
print()
print("Next steps:")
print("  1. Open and run: 01_evaluation_framework.ipynb")
print("  2. Open and run: 02_baseline_models.ipynb")
print("  3. Open and run: Hull_Tactical_Solution_Presentation.ipynb")
print("  4. Review: FIXES_APPLIED.md for detailed documentation")
print()
