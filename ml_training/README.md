# ML Training Scripts for Cognitive Triad

This directory contains standalone training scripts for all machine learning models used in the Cognitive Triad platform.

## 📁 Contents

- `train_rca_models.py` - Root Cause Analysis model training
- `train_forecasting_models.py` - Time series forecasting model training
- `README.md` - This file

## 🚀 Quick Start

### Prerequisites

```bash
# Ensure you're in the backend directory
cd /app/backend

# All dependencies should already be installed, but if needed:
pip install -r requirements.txt
```

### Train RCA Models

```bash
cd /app/ml_training
python3 train_rca_models.py
```

**This will:**
- Generate 1000 synthetic IT incident records
- Train 3 models: Decision Tree, Random Forest, Neural Network
- Perform cross-validation
- Generate confusion matrices
- Save models to `/app/backend/ml_models/`

**Output files:**
- `decision_tree.pkl`
- `random_forest.pkl`
- `neural_network.pkl`
- `le_category.pkl`, `le_priority.pkl`, `le_system.pkl`, `le_cause.pkl` (encoders)
- Visualizations: confusion matrices, feature importance, model comparison

### Train Forecasting Models

```bash
cd /app/ml_training
python3 train_forecasting_models.py
```

**This will:**
- Generate 731 days (2023-2024) of realistic ticket data
- Perform time series analysis
- Train SARIMA and LSTM models
- Evaluate with MAE, RMSE, MAPE
- Save models to `/app/backend/ml_models/`

**Output files:**
- `ticket_history.csv` (731 days of data)
- `sarima_model.pkl`
- `lstm_model.h5`
- `lstm_scaler.pkl`
- Visualizations: time series plots, forecasts, training curves

## 📊 Model Details

### Root Cause Analysis (RCA)

**Models:**
1. **Decision Tree**
   - Max depth: 10
   - Min samples split: 10
   - Fast inference, interpretable

2. **Random Forest** (Recommended)
   - 100 trees
   - Max depth: 10
   - Best accuracy, feature importance

3. **Neural Network**
   - Architecture: 64→32→16
   - Activation: ReLU
   - Early stopping enabled

**Features:**
- Category (Network, Database, Application, Hardware, Security)
- Priority (Low, Medium, High, Critical)
- Affected System (Web Server, DB Server, API Gateway, etc.)
- Error Code (100-999)

**Target:**
- Root Cause (8 classes: Configuration Error, Resource Exhaustion, etc.)

### Time Series Forecasting

**Models:**
1. **SARIMA (Seasonal ARIMA)**
   - Order: (1,1,1)
   - Seasonal Order: (1,1,1,7) - Weekly seasonality
   - Good for interpretability
   - Captures weekly patterns

2. **LSTM (Deep Learning)**
   - 3 LSTM layers (50 units each)
   - Dropout (0.2) for regularization
   - Look-back window: 30 days
   - Better for complex patterns

**Temporal Patterns in Data:**
- Business hours peak (9am-5pm)
- Weekend reduction (40% of weekday)
- Monday spikes (+10 tickets)
- Month-end peaks (+5 tickets)
- Quarterly spikes (+8 tickets)
- Random incident spikes (5% chance)
- Growth trend over time

## 🔧 Customization

### Modify Data Generation

**RCA Models (`train_rca_models.py`):**

```python
# Change number of samples
df = generate_incident_data(n_samples=5000)

# Modify categories/priorities
categories = ['Your', 'Custom', 'Categories']
priorities = ['Your', 'Priorities']
```

**Forecasting Models (`train_forecasting_models.py`):**

```python
# Change date range
df = generate_realistic_ticket_data(
    start_date='2020-01-01', 
    end_date='2025-12-31'
)

# Adjust temporal patterns in generate_realistic_ticket_data()
```

### Modify Model Hyperparameters

**Decision Tree:**
```python
model = DecisionTreeClassifier(
    max_depth=15,  # Increase depth
    min_samples_split=5,
    random_state=42
)
```

**Random Forest:**
```python
model = RandomForestClassifier(
    n_estimators=200,  # More trees
    max_depth=15,
    random_state=42
)
```

**LSTM:**
```python
model = Sequential([
    LSTM(100, return_sequences=True),  # More units
    LSTM(100),
    Dense(1)
])
```

## 📈 Evaluation Metrics

### Classification (RCA)
- **Accuracy**: Overall correct predictions
- **Precision/Recall/F1**: Per-class performance
- **Cross-validation**: 5-fold CV scores
- **Confusion Matrix**: Visual error analysis

### Forecasting
- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error): Percentage error

## 🎯 Using Trained Models

### Load and Use RCA Model

```python
import pickle
import numpy as np

# Load model and encoders
rf_model = pickle.load(open('/app/backend/ml_models/random_forest.pkl', 'rb'))
le_cat = pickle.load(open('/app/backend/ml_models/le_category.pkl', 'rb'))
le_pri = pickle.load(open('/app/backend/ml_models/le_priority.pkl', 'rb'))
le_sys = pickle.load(open('/app/backend/ml_models/le_system.pkl', 'rb'))
le_cause = pickle.load(open('/app/backend/ml_models/le_cause.pkl', 'rb'))

# Prepare input
features = np.array([[
    le_cat.transform(['Network'])[0],
    le_pri.transform(['High'])[0],
    le_sys.transform(['Web Server'])[0],
    500  # error code
]])

# Predict
prediction = rf_model.predict(features)[0]
probability = rf_model.predict_proba(features)[0].max()

root_cause = le_cause.inverse_transform([prediction])[0]
print(f"Predicted: {root_cause} (confidence: {probability:.2%})")
```

### Load and Use Forecasting Model

```python
import pickle
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load data and model
df = pd.read_csv('/app/backend/ml_models/ticket_history.csv')
df['date'] = pd.to_datetime(df['date'])
ts = df.set_index('date')['count']

sarima_model = pickle.load(open('/app/backend/ml_models/sarima_model.pkl', 'rb'))

# Forecast next 30 days
forecast = sarima_model.forecast(steps=30)
print(forecast)
```

## 📝 Training Logs

Both scripts provide detailed logs including:
- Data generation statistics
- Model architecture
- Training progress
- Evaluation metrics
- File locations

Example output:
```
======================================================================
RCA MODEL TRAINING PIPELINE
======================================================================
Generating 1000 incident records...
✓ Generated 1000 incidents
  Categories: 5
  Root Causes: 8

Preprocessing data...
✓ Feature matrix: (1000, 4)
✓ Target classes: 8

==================================================
TRAINING RANDOM FOREST
==================================================
Accuracy: 0.8650
Cross-validation scores: [0.845, 0.870, 0.860, 0.875, 0.855]
Mean CV score: 0.8610 (+/- 0.0218)

Feature Importances:
  Category: 0.3245
  Priority: 0.2156
  System: 0.2890
  Error Code: 0.1709
```

## 🐛 Troubleshooting

**Issue: Models not loading in application**
- Ensure models are saved to `/app/backend/ml_models/`
- Restart backend: `sudo supervisorctl restart backend`

**Issue: LSTM training slow**
- Reduce epochs: `epochs=30`
- Reduce batch size: `batch_size=64`
- Use GPU if available

**Issue: Poor model accuracy**
- Increase training data: `n_samples=5000`
- Tune hyperparameters
- Check data quality and correlations

## 📚 Further Reading

- **SARIMA**: [Statsmodels Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
- **LSTM**: [Keras LSTM Guide](https://keras.io/api/layers/recurrent_layers/lstm/)
- **Random Forest**: [Scikit-learn RF](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## 💡 Tips

1. **Always retrain models** when:
   - New data patterns emerge
   - Business requirements change
   - Model performance degrades

2. **Monitor model drift**:
   - Track accuracy over time
   - Compare predictions vs actuals
   - Retrain periodically

3. **Experiment with features**:
   - Add new incident attributes
   - Engineer temporal features
   - Test different encodings

4. **Save training metadata**:
   - Model version
   - Training date
   - Hyperparameters
   - Performance metrics

## 🎓 Learning Resources

These scripts are designed to be educational. Key concepts demonstrated:

- Time series decomposition
- Cross-validation techniques
- Neural network architecture
- Feature engineering
- Model evaluation
- Hyperparameter tuning
- Data preprocessing
- Visualization techniques

Feel free to modify and experiment!
