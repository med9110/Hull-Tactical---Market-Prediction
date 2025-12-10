# 🎯 Hull Tactical - Quick Reference Cheatsheet

## 🚨 CRITICAL WARNINGS

### ❌ DO NOT USE (Leakage Models):
- **Model_1, 4, 5, 6, 7** from `hull-tactical-ensemble-of-solutions.ipynb`
- Any code using `true_targets` dict
- Any code with `scipy.optimize` on test set
- Public LB scores > 10 are **ALL LEAKAGE**

### ✅ SAFE TO USE:
- ElasticNet from `hull-starter-notebook.ipynb`
- Model_2 (uses train data properly)
- Model_3 (stacking ensemble)
- EDA framework from `htmp-eda-which-makes-sense.ipynb`

---

## 📊 Competition Overview

| Aspect | Details |
|--------|---------|
| **Task** | Predict S&P 500 returns → convert to allocation [0, 2] |
| **Metric** | Volatility-adjusted Sharpe ratio |
| **Penalty 1** | Strategy vol > 1.2× market vol |
| **Penalty 2** | Strategy return < market return |
| **Public LB** | ❌ MEANINGLESS (last 180 train days) |
| **What matters** | Private leaderboard (unseen future data) |

---

## 🎯 Target Scores

| Model | CV Score (trust this) | Public LB (ignore) |
|-------|----------------------|-------------------|
| Constant 0.8 | 0.45 | 0.66 |
| ElasticNet | 0.48-0.52 | 0.70+ |
| LightGBM | 0.52-0.58 | 0.75+ |
| Ensemble | 0.55-0.65 | 0.80+ |
| **Your goal** | **> 0.55** | (don't care) |

---

## 📁 Files Created for You

```
challenge enigmes/
├── 01_evaluation_framework.ipynb  ← Start here (metric + CV)
├── 02_baseline_models.ipynb       ← ElasticNet + LightGBM
├── 03_submission_template.ipynb   ← Final submission
├── STEP_BY_STEP_GUIDE.md          ← Full guide (read this!)
└── QUICK_REFERENCE.md             ← This file
```

---

## ⚡ Quick Start (5 Steps)

### Step 1: Run Evaluation Framework
```python
# Open: 01_evaluation_framework.ipynb
# Run all cells
# Expected: Constant 0.8 → CV ~0.45
```

### Step 2: Train ElasticNet
```python
# Open: 02_baseline_models.ipynb
# Run ElasticNet section
# Tune SIGNAL_MULTIPLIER: [200, 300, 400, 500, 600]
# Target: CV > 0.50
```

### Step 3: Add LightGBM
```python
# Continue in: 02_baseline_models.ipynb
# Run LightGBM section
# Target: CV > 0.55
```

### Step 4: Ensemble
```python
# Combine ElasticNet + LightGBM
# Tune weights: e.g., [0.3, 0.7]
# Target: CV > 0.58
```

### Step 5: Submit
```python
# Open: 03_submission_template.ipynb
# Copy your best model code
# Hard-code tuned hyperparameters
# Test locally, then submit
```

---

## 🔧 Essential Code Snippets

### Load Data
```python
import pandas as pd
train = pd.read_csv('./hull-tactical-market-prediction/train.csv')
train = train[train['date_id'] >= 1000]  # Trim sparse dates
```

### Feature Columns
```python
feature_cols = [c for c in train.columns 
                if c.startswith(('M', 'E', 'I', 'P', 'V', 'S', 'MOM', 'D'))]
```

### Target Column
```python
target = 'market_forward_excess_returns'  # Use this (normalized)
# NOT 'forward_returns' (raw)
```

### Time-Series CV
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=10, test_size=180)
for train_idx, test_idx in tscv.split(train):
    # Train on train_idx, test on test_idx
    pass
```

### Convert Return → Signal
```python
def convert_return_to_signal(pred_return, multiplier=400):
    signal = pred_return * multiplier + 1.0
    return np.clip(signal, 0.0, 2.0)
```

### ElasticNet Template
```python
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

model = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000))
])

X = train[feature_cols]
y = train['market_forward_excess_returns']
mask = y.notna()
model.fit(X[mask], y[mask])
```

### LightGBM Template
```python
import lightgbm as lgb

X = train[feature_cols].fillna(-999)
y = train['market_forward_excess_returns']
mask = y.notna()

train_set = lgb.Dataset(X[mask], label=y[mask])

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,
    'num_leaves': 63,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

model = lgb.train(params, train_set, num_boost_round=3000)
```

### Kaggle predict() Function
```python
import polars as pl

def predict(test: pl.DataFrame) -> float:
    global MODEL, FITTED
    
    if not FITTED:
        train_models()  # Train once on first call
        FITTED = True
    
    test_pd = test.to_pandas()
    X_test = test_pd[feature_cols]
    
    pred_return = MODEL.predict(X_test)
    signal = convert_return_to_signal(pred_return, multiplier=400)
    
    if len(signal) == 1:
        return float(signal[0])
    return signal.astype(np.float64)
```

---

## 🎓 Hyperparameters to Tune

### ElasticNet
```python
alpha: [0.01, 0.1, 1.0, 10.0]
l1_ratio: [0.3, 0.5, 0.7, 0.9]
```

### LightGBM
```python
learning_rate: [0.01, 0.05]
num_leaves: [31, 63, 127]
max_depth: [6, 8, 10]
feature_fraction: [0.7, 0.8, 0.9]
```

### Signal Conversion
```python
SIGNAL_MULTIPLIER: [200, 300, 400, 500, 600]
```
**Rule:** Higher = more aggressive, lower = more conservative.

### Ensemble Weights
```python
# If LGBM better than ElasticNet
weights = [0.2, 0.8]  # [ENET, LGBM]

# If similar performance
weights = [0.5, 0.5]
```

---

## 📊 Metric Breakdown

### Portfolio Score Formula
```python
# 1. Calculate strategy returns
strategy_returns = risk_free_rate * (1 - position) + position * forward_returns

# 2. Calculate excess returns
strategy_excess = strategy_returns - risk_free_rate

# 3. Calculate Sharpe
strategy_mean = (1 + strategy_excess).prod() ** (1/n) - 1
strategy_std = strategy_returns.std()
sharpe = strategy_mean / strategy_std * sqrt(252)

# 4. Calculate penalties
vol_penalty = 1 + max(0, strategy_vol / market_vol - 1.2)
return_penalty = 1 + (max(0, market_return - strategy_return) * 252) ** 2 / 100

# 5. Adjusted Sharpe
adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
```

### What This Means
- **Maximize Sharpe:** High returns, low volatility
- **Stay under 1.2× market vol:** Use your volatility budget
- **Don't underperform market:** Or get penalized quadratically

---

## ⏰ Timeline

| Date | Deadline | Action |
|------|----------|--------|
| **Dec 8** | Entry & Team Merger | Accept rules, form team |
| **Dec 15** | Final Submission | Submit notebook |
| **Jun 16, 2026** | Competition End | Forecasting phase results |

---

## 🚦 Submission Checklist

Before submitting:
- [ ] Notebook runs in < 8 hours
- [ ] No leakage (no `true_targets`, no test data in training)
- [ ] `predict()` function works with Polars input
- [ ] Signal multiplier tuned on CV (not LB)
- [ ] Tested locally first
- [ ] All imports available on Kaggle

---

## 📞 Help & Resources

### Kaggle Discussion Topics to Search:
- "Proper cross validation"
- "Metric explanation"
- "Signal conversion"
- "Feature engineering"

### Avoid Topics:
- "LB 17.xxx" (leakage)
- "Optimal predictions" (leakage)
- "Perfect score" (leakage)

---

## 🎯 Your Daily TODO

### Today:
1. ✅ Read `STEP_BY_STEP_GUIDE.md` (full guide)
2. ✅ Run `01_evaluation_framework.ipynb`
3. ✅ Understand metric and baseline scores

### Tomorrow:
1. Run `02_baseline_models.ipynb` (ElasticNet section)
2. Tune `SIGNAL_MULTIPLIER`
3. Record CV score

### Day 3:
1. Add LightGBM to `02_baseline_models.ipynb`
2. Compare with ElasticNet
3. Target: CV > 0.52

### Day 4:
1. Build ensemble
2. Tune weights
3. Target: CV > 0.55

### Day 5+:
1. Feature engineering
2. Refinement
3. Prepare submission

---

## 💡 Pro Tips

1. **Trust CV, not LB**
   - Public LB = leakage
   - Your CV = reality

2. **Start simple**
   - Get ElasticNet working first
   - Then add complexity

3. **Log everything**
   ```python
   results = {
       'date': '2025-11-26',
       'model': 'LGBMv2',
       'cv_score': 0.56,
       'params': {...}
   }
   ```

4. **Version control**
   - Save notebook versions
   - Track what works

5. **Test locally**
   ```python
   inference_server.run_local_gateway(...)
   ```

---

## 🏆 Success Criteria

| Level | CV Score | Status |
|-------|----------|--------|
| Beginner | > 0.50 | Beat baselines |
| Intermediate | > 0.55 | Competitive |
| Advanced | > 0.60 | Top tier |
| Expert | > 0.65 | Podium potential |

**Your goal: Get to Intermediate (0.55+) by Dec 8, then push for Advanced.**

---

## 🚀 Let's Go!

1. Open `01_evaluation_framework.ipynb`
2. Run it
3. Come back with questions

**Remember: The competition starts AFTER Dec 15, not before!**

Good luck! 🎯
