# 🎯 Hull Tactical Market Prediction - Complete Step-by-Step Guide

## 📋 Executive Summary

You're competing in a **time-series forecasting competition** to predict S&P 500 returns while managing volatility. The goal is NOT just prediction accuracy - it's maximizing risk-adjusted returns (Sharpe ratio) within volatility constraints.

**Key Competition Facts:**
- **Public LB is MEANINGLESS** - test data = last 180 days of train
- Many top public notebooks use **leakage** (looking up future prices)
- Real competition starts in **forecasting phase** (June 2026)
- Your models will run on **unseen future market data**

---

## ⚠️ What You Have (Analysis of Your Notebooks)

### ✅ **Good Notebooks (USE THESE):**

1. **`htmp-eda-which-makes-sense.ipynb`**
   - ✅ Proper metric implementation
   - ✅ Correct time-series CV
   - ✅ Great baseline strategies
   - **Action:** Use this as your validation framework

2. **`hull-starter-notebook.ipynb`**
   - ✅ Clean ElasticNet baseline
   - ✅ Proper feature engineering
   - ✅ No leakage
   - **Action:** This is your starting point for models

### ❌ **Dangerous Notebooks (LEARN BUT DON'T COPY):**

3. **`hull-tactical-ensemble-of-solutions.ipynb`**
   - ❌ Model_1: Uses `true_targets` - **pure leakage**
   - ❌ Model_4, 5, 6: Same leakage pattern
   - ❌ Model_7: `scipy.optimize` on 180 test days - **overfitting**
   - ✅ Model_2: Uses train data properly
   - ✅ Model_3: Stacking ensemble (legit)
   - **Action:** Only use Models 2 & 3 logic, ignore the rest

---

## 🗓️ Your Week-by-Week Action Plan

### **Week 1: Foundation (Days 1-7)**

#### **Day 1-2: Setup & Understanding**
1. Read competition overview thoroughly
2. Understand the metric:
   - Base: Sharpe ratio
   - Penalty 1: If strategy vol > 1.2× market vol
   - Penalty 2: If strategy underperforms market
3. Run **`01_evaluation_framework.ipynb`**:
   ```bash
   # Test constant allocations
   # Understand baseline scores
   ```

#### **Day 3-4: Data Exploration**
1. Load train/test data
2. Check missingness over time:
   - Early dates (< 1000) are very sparse
   - Decide on cutoff date
3. Plot target variable:
   ```python
   plt.scatter(train['date_id'], train['market_forward_excess_returns'])
   ```
4. Identify volatile periods (2008, 2020)

#### **Day 5-7: First Model**
1. Run **`02_baseline_models.ipynb`**
2. Train ElasticNet with proper CV
3. Tune `SIGNAL_MULTIPLIER`:
   ```python
   for mult in [200, 300, 400, 500, 600]:
       # Test each, pick best CV score
   ```
4. **Target: Beat constant allocation (0.45-0.50 score)**

---

### **Week 2: Model Improvement (Days 8-14)**

#### **Day 8-10: Add LightGBM**
1. Implement LightGBM model
2. Tune hyperparameters:
   - `learning_rate`: [0.01, 0.05]
   - `num_leaves`: [31, 63, 127]
   - `max_depth`: [6, 8, 10]
3. Cross-validate properly
4. **Target: Improve over ElasticNet by 5-10%**

#### **Day 11-12: Feature Engineering**
1. Create lag features:
   ```python
   train['lagged_return_1'] = train['forward_returns'].shift(1)
   train['lagged_return_5'] = train['forward_returns'].shift(5)
   ```
2. Rolling statistics:
   ```python
   train['vol_20d'] = train['forward_returns'].rolling(20).std()
   train['mean_20d'] = train['forward_returns'].rolling(20).mean()
   ```
3. Feature interactions:
   ```python
   train['momentum_value'] = train['MOM1'] * train['P10']
   ```

#### **Day 13-14: Ensemble**
1. Build simple weighted average:
   ```python
   pred = 0.3 * elasticnet + 0.7 * lgbm
   ```
2. Grid search weights on CV folds
3. **Target: Best single model score + 3-5%**

---

### **Week 3: Refinement (Days 15-20)**

#### **Day 15-16: Signal Mapping Optimization**
Current mapping is **linear**:
```python
signal = predicted_return * 400 + 1.0
```

Try **regime-aware** mapping:
```python
if current_volatility > high_threshold:
    # Reduce exposure in high-vol regime
    signal = predicted_return * 200 + 0.8
else:
    signal = predicted_return * 500 + 1.0
```

#### **Day 17-18: Volatility Targeting**
Implement strategy to stay just under 1.2× market vol:
```python
# Estimate strategy vol
strategy_vol = estimate_vol(predictions)
market_vol = train['forward_returns'].std() * np.sqrt(252)

# Scale down if needed
if strategy_vol / market_vol > 1.19:
    predictions *= (1.19 * market_vol / strategy_vol)
```

#### **Day 19-20: Final Validation**
1. Run full CV (10+ folds)
2. Check fold stability:
   - Are scores consistent?
   - Any folds with huge outliers?
3. Test on different time periods
4. **Target: CV score > 0.6**

---

### **Week 4: Submission Prep (Days 21+)**

#### **Day 21-22: Clean Submission Notebook**
1. Use **`03_submission_template.ipynb`**
2. Copy your best model code
3. Hard-code tuned hyperparameters
4. Remove all debugging/prints

#### **Day 23: Local Testing**
```python
# Test predict() function locally
test_batch = test.head(1)
result = predict(test_batch)
assert 0.0 <= result <= 2.0
```

#### **Day 24: Submit & Monitor**
1. Commit notebook on Kaggle
2. Click "Submit to Competition"
3. Watch for errors
4. **Ignore public LB score** (it's meaningless)

---

## 🔧 Critical Code Patterns

### **1. Proper Predict Function Structure**
```python
def predict(test: pl.DataFrame) -> float:
    global MODEL, FITTED
    
    if not FITTED:
        # Train on first call
        train_models()
        FITTED = True
    
    # Convert polars → pandas
    test_pd = test.to_pandas()
    
    # Extract features (same as training!)
    X_test = test_pd[FEATURE_COLS]
    
    # Predict returns
    pred_returns = MODEL.predict(X_test)
    
    # Convert to signal [0, 2]
    signal = pred_returns * SIGNAL_MULTIPLIER + 1.0
    signal = np.clip(signal, 0.0, 2.0)
    
    # Return scalar if single sample
    if len(signal) == 1:
        return float(signal[0])
    return signal
```

### **2. Time-Series CV Pattern**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=10, test_size=180)

for fold, (train_idx, test_idx) in enumerate(tscv.split(train)):
    X_train = train.iloc[train_idx]
    X_test = train.iloc[test_idx]
    
    # Train model ONLY on train_idx
    model.fit(X_train)
    
    # Predict on test_idx
    preds = model.predict(X_test)
    
    # Score
    score = portfolio_score(X_test[['forward_returns', 'risk_free_rate']], 
                           pd.DataFrame({'prediction': preds}))
    
    print(f"Fold {fold}: {score:.4f}")
```

### **3. Feature Imputation**
```python
# Option 1: Forward fill (for slow-moving features like E*, I*)
train[macro_features] = train[macro_features].fillna(method='ffill')

# Option 2: Mean imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_train)

# Option 3: LightGBM handles NaN natively
train[features].fillna(-999)  # Use sentinel value
```

---

## 🎓 Key Learnings from Public Notebooks

### **From `htmp-eda-which-makes-sense.ipynb`:**
✅ **Volatility penalty formula:**
```python
excess_vol = max(0, strategy_vol / market_vol - 1.2)
vol_penalty = 1 + excess_vol
```
→ You have a 20% volatility budget! Use it wisely.

✅ **Constant 0.8 allocation scores ~0.45:**
→ Your models must beat this or they're useless.

✅ **Day-of-week effects might exist:**
→ Wednesday/Thursday returns are lower (but weak signal).

### **From `hull-starter-notebook.ipynb`:**
✅ **Good feature subset:**
```python
vars_to_keep = ["S2", "E2", "E3", "P9", "S1", "S5", "I2", "P8",
                "P10", "P12", "P13", "U1", "U2"]
```

✅ **Signal multiplier of 400 is a good starting point:**
```python
signal = predicted_return * 400 + 1
```

### **From `hull-tactical-ensemble-of-solutions.ipynb`:**
❌ **What NOT to do:**
```python
# DON'T DO THIS - it's cheating!
true_targets = dict(zip(train['date_id'], train['forward_returns']))
pred = true_targets.get(date_id)  # ← LEAKAGE!
```

✅ **What to learn:**
- Model_3's stacking approach (CatBoost + XGB + LGBM + Ridge)
- Ensemble weighting strategies

---

## 📊 Expected Scores

| Strategy | Public LB | CV (proper) | Private (expected) |
|----------|-----------|-------------|-------------------|
| Constant 0.8 | 0.66 | 0.45 | 0.45 |
| Constant 1.0 | 0.50 | 0.42 | 0.42 |
| ElasticNet | 0.70+ | 0.48-0.52 | 0.48-0.52 |
| LightGBM | 0.75+ | 0.52-0.58 | 0.52-0.58 |
| Ensemble | 0.80+ | 0.55-0.65 | 0.55-0.65 |
| **Leakage models** | **10-17** | **N/A** | **0** |

**Trust your CV, ignore public LB!**

---

## ⚡ Quick Wins

### **Win #1: Drop Early Sparse Data**
```python
train = train[train['date_id'] >= 1000]
```
→ Improves model quality, reduces noise.

### **Win #2: Use `market_forward_excess_returns` as Target**
Already normalized and outlier-clipped:
```python
y = train['market_forward_excess_returns']
```
→ Easier to model than raw `forward_returns`.

### **Win #3: Tune Signal Multiplier per Model**
ElasticNet and LightGBM may need different multipliers:
```python
signal_enet = pred_enet * 350 + 1
signal_lgbm = pred_lgbm * 450 + 1
```

### **Win #4: Weight Ensemble Toward Better Model**
If LightGBM CV = 0.55, ElasticNet CV = 0.50:
```python
ensemble = 0.3 * enet + 0.7 * lgbm  # Favor LGBM
```

---

## 🚨 Common Mistakes to Avoid

### ❌ **Mistake 1: Using Test Data for Training**
```python
# BAD
all_data = pd.concat([train, test])
model.fit(all_data)  # ← test has no targets!
```

### ❌ **Mistake 2: Random K-Fold**
```python
# BAD
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)  # ← breaks time order!
```
Use `TimeSeriesSplit` instead.

### ❌ **Mistake 3: Looking at Public LB**
Public LB = last 180 days of train = **total leakage**.
**Solution:** Trust your CV, ignore LB.

### ❌ **Mistake 4: Over-Optimizing Signal Multiplier**
Tuning multiplier from 400 → 401 based on 0.001 CV improvement = **overfitting**.
**Solution:** Grid search [200, 300, 400, 500, 600], pick robust value.

### ❌ **Mistake 5: Forgetting Polars → Pandas Conversion**
```python
# Kaggle API gives you Polars
def predict(test: pl.DataFrame) -> float:
    test_pd = test.to_pandas()  # ← DON'T FORGET THIS
```

---

## 📦 Files I Created for You

1. **`01_evaluation_framework.ipynb`**
   - Metric implementation
   - Time-series CV
   - Baseline tests

2. **`02_baseline_models.ipynb`**
   - ElasticNet
   - LightGBM
   - Simple ensemble

3. **`03_submission_template.ipynb`**
   - Clean submission code
   - No leakage
   - Kaggle API integration

---

## 🎯 Your Target Timeline

| Date | Milestone |
|------|-----------|
| **Dec 1** | Complete Week 1 (foundation) |
| **Dec 8** | Complete Week 2 (LightGBM + ensemble) |
| **Dec 8** | **ENTRY DEADLINE** - must accept rules |
| **Dec 8** | **TEAM MERGER DEADLINE** |
| **Dec 12** | Complete Week 3 (refinement) |
| **Dec 15** | **FINAL SUBMISSION DEADLINE** |

---

## 💡 Pro Tips

1. **Start simple, iterate fast:**
   - Day 1: Run evaluation framework
   - Day 2: ElasticNet baseline
   - Day 3: Add LightGBM
   - Day 4+: Improve incrementally

2. **Log everything:**
   ```python
   results = {
       'model': 'LightGBM_v3',
       'cv_score': 0.56,
       'signal_mult': 450,
       'features': feature_cols,
       'notes': 'Added rolling vol features'
   }
   # Save to CSV/JSON
   ```

3. **Version your notebooks:**
   - `02_baseline_models_v1.ipynb`
   - `02_baseline_models_v2_added_lgbm.ipynb`
   - etc.

4. **Test locally before submitting:**
   ```python
   inference_server.run_local_gateway(('/kaggle/input/...',))
   ```

5. **Read discussions:**
   - Kaggle discussion forum has gold
   - Look for "proper CV" threads
   - Avoid "LB 17.xxx" threads (leakage)

---

## 🏆 Final Checklist Before Submission

- [ ] Models trained on `date_id >= 1000` only
- [ ] No usage of `true_targets` or train data in `predict()`
- [ ] Signal multiplier tuned on CV (not LB)
- [ ] Ensemble weights tuned on CV (not LB)
- [ ] Tested locally with `run_local_gateway`
- [ ] Notebook runs in < 8 hours
- [ ] `predict()` returns scalar for single sample
- [ ] All imports are valid Kaggle packages
- [ ] No internet access required

---

## 📞 Next Steps

1. **TODAY:** Run `01_evaluation_framework.ipynb`
2. **DAY 2:** Run `02_baseline_models.ipynb` with ElasticNet
3. **DAY 3:** Add LightGBM to `02_baseline_models.ipynb`
4. **DAY 4:** Tune signal multipliers
5. **DAY 5:** Build ensemble
6. **DAY 6-7:** Feature engineering
7. **DAY 8+:** Refinement and submission

---

## 🤝 Good Luck!

Remember:
- **Public LB doesn't matter**
- **Trust your CV**
- **Simple models often win**
- **Avoid leakage at all costs**

You've got this! 🚀
