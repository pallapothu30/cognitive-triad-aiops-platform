"""
Ticket Load Forecasting Model Training Script
==============================================
Trains SARIMA and LSTM models for IT service request volume prediction
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time series libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Deep learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ==================== DATA GENERATION ====================
def generate_realistic_ticket_data(start_date='2023-01-01', end_date='2024-12-31'):
    """
    Generate realistic IT ticket data with temporal patterns
    
    Returns:
        DataFrame with datetime and ticket counts
    """
    print("Generating realistic ticket data...")
    
    # Generate hourly timestamps
    dates = pd.date_range(start=start_date, end=end_date, freq='h')
    ticket_counts = []
    
    for dt in dates:
        # Base load
        base = 5
        
        # Business hours effect (9am-5pm)
        hour = dt.hour
        if 9 <= hour <= 17:
            base += 15
        elif 8 <= hour <= 18:
            base += 8
        elif 6 <= hour <= 20:
            base += 3
        
        # Weekday vs weekend
        if dt.weekday() >= 5:  # Weekend
            base = base * 0.4
        
        # Monday spike (more issues after weekend)
        if dt.weekday() == 0 and 9 <= hour <= 12:
            base += 10
        
        # Month-end spike (deployments, reporting)
        if dt.day >= 28:
            base += 5
        
        # Quarterly patterns (Q-end maintenance)
        if dt.month in [3, 6, 9, 12] and dt.day >= 25:
            base += 8
        
        # Random incidents and noise
        noise = np.random.poisson(2)
        spike = np.random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 15], 
                                p=[0.95, 0, 0, 0, 0, 0, 0, 0, 0, 0.05])
        
        # Trend (gradual increase over time)
        trend = (dt - dates[0]).days * 0.01
        
        count = int(max(0, base + noise + spike + trend))
        ticket_counts.append(count)
    
    ts_data = pd.DataFrame({'datetime': dates, 'count': ticket_counts})
    
    # Aggregate to daily
    ts_data['date'] = ts_data['datetime'].dt.date
    daily_data = ts_data.groupby('date')['count'].sum().reset_index()
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    
    print(f"✓ Generated {len(daily_data)} days of ticket data")
    print(f"  Date range: {daily_data['date'].min()} to {daily_data['date'].max()}")
    print(f"  Average tickets/day: {daily_data['count'].mean():.2f}")
    print(f"  Min/Max: {daily_data['count'].min()} / {daily_data['count'].max()}")
    
    return daily_data

# ==================== EXPLORATORY ANALYSIS ====================
def analyze_time_series(df, save_dir='/app/ml_training/'):
    """Perform exploratory time series analysis"""
    print("\n" + "="*50)
    print("TIME SERIES ANALYSIS")
    print("="*50)
    
    ts = df.set_index('date')['count']
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(f"  Mean: {ts.mean():.2f}")
    print(f"  Std: {ts.std():.2f}")
    print(f"  Min: {ts.min()}")
    print(f"  Max: {ts.max()}")
    
    # Plot time series
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Original series
    axes[0].plot(ts.index, ts.values, linewidth=0.8)
    axes[0].set_title('Daily Ticket Counts')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    # 7-day moving average
    ma7 = ts.rolling(window=7).mean()
    axes[1].plot(ts.index, ts.values, alpha=0.3, label='Original')
    axes[1].plot(ma7.index, ma7.values, linewidth=2, label='7-day MA')
    axes[1].set_title('7-Day Moving Average')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 30-day moving average
    ma30 = ts.rolling(window=30).mean()
    axes[2].plot(ts.index, ts.values, alpha=0.3, label='Original')
    axes[2].plot(ma30.index, ma30.values, linewidth=2, label='30-day MA', color='red')
    axes[2].set_title('30-Day Moving Average')
    axes[2].set_ylabel('Count')
    axes[2].set_xlabel('Date')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir + 'time_series_overview.png', dpi=100)
    plt.close()
    print(f"  ✓ Saved time series overview")
    
    # Seasonal decomposition
    print("\nPerforming seasonal decomposition...")
    decomposition = seasonal_decompose(ts, model='additive', period=7)
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal (7-day)')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir + 'seasonal_decomposition.png', dpi=100)
    plt.close()
    print(f"  ✓ Saved seasonal decomposition")
    
    # ACF and PACF
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    plot_acf(ts, lags=40, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF)')
    plot_pacf(ts, lags=40, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    
    plt.tight_layout()
    plt.savefig(save_dir + 'acf_pacf.png', dpi=100)
    plt.close()
    print(f"  ✓ Saved ACF/PACF plots")

# ==================== SARIMA MODEL ====================
def train_sarima_model(df, order=(1,1,1), seasonal_order=(1,1,1,7), forecast_days=30):
    """Train SARIMA model for time series forecasting"""
    print("\n" + "="*50)
    print("TRAINING SARIMA MODEL")
    print("="*50)
    
    ts = df.set_index('date')['count']
    
    # Split into train/test
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]
    
    print(f"Train set: {len(train)} days")
    print(f"Test set: {len(test)} days")
    
    # Train SARIMA model
    print(f"\nTraining SARIMA{order}x{seasonal_order}...")
    model = SARIMAX(train, 
                    order=order, 
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    
    fitted_model = model.fit(disp=False, maxiter=200)
    
    print("✓ Model trained successfully")
    print(f"\nModel Summary:")
    print(f"  AIC: {fitted_model.aic:.2f}")
    print(f"  BIC: {fitted_model.bic:.2f}")
    
    # Forecast on test set
    forecast = fitted_model.forecast(steps=len(test))
    
    # Calculate metrics
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test)) * 100
    
    print(f"\nTest Set Performance:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Plot predictions
    plt.figure(figsize=(15, 6))
    plt.plot(train.index, train.values, label='Train', linewidth=0.8)
    plt.plot(test.index, test.values, label='Test', linewidth=0.8)
    plt.plot(test.index, forecast, label='Forecast', linewidth=2, linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Ticket Count')
    plt.title('SARIMA Model - Train/Test/Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/app/ml_training/sarima_forecast.png', dpi=100)
    plt.close()
    print(f"  ✓ Saved forecast plot")
    
    # Future forecast
    future_forecast = fitted_model.forecast(steps=forecast_days)
    
    return fitted_model, {'mae': mae, 'rmse': rmse, 'mape': mape}

# ==================== LSTM MODEL ====================
def prepare_lstm_data(data, look_back=30):
    """Prepare data for LSTM model"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def train_lstm_model(df, look_back=30, epochs=50, batch_size=32):
    """Train LSTM model for time series forecasting"""
    print("\n" + "="*50)
    print("TRAINING LSTM MODEL")
    print("="*50)
    
    ts = df['count']
    
    # Prepare data
    print(f"Preparing data with look_back={look_back}...")
    X, y, scaler = prepare_lstm_data(ts, look_back)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Build LSTM model
    print("\nBuilding LSTM architecture...")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print(model.summary())
    
    # Train model
    print(f"\nTraining for {epochs} epochs...")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("✓ Model trained successfully")
    
    # Evaluate on test set
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
    
    print(f"\nTest Set Performance:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_title('Model MAE')
    axes[1].set_ylabel('MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/app/ml_training/lstm_training_history.png', dpi=100)
    plt.close()
    print(f"  ✓ Saved training history")
    
    # Plot predictions
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_actual, label='Actual', linewidth=1)
    plt.plot(y_pred, label='Predicted', linewidth=1, alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel('Ticket Count')
    plt.title('LSTM Model - Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/app/ml_training/lstm_predictions.png', dpi=100)
    plt.close()
    print(f"  ✓ Saved predictions plot")
    
    return model, scaler, {'mae': mae, 'rmse': rmse, 'mape': mape}

# ==================== MAIN TRAINING PIPELINE ====================
def main():
    print("="*70)
    print("FORECASTING MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Generate data
    df = generate_realistic_ticket_data()
    
    # Save to CSV
    df.to_csv('/app/backend/ml_models/ticket_history.csv', index=False)
    print(f"\n✓ Saved data to /app/backend/ml_models/ticket_history.csv")
    
    # Step 2: Exploratory analysis
    analyze_time_series(df)
    
    # Step 3: Train SARIMA
    sarima_model, sarima_metrics = train_sarima_model(df)
    
    # Save SARIMA model
    pickle.dump(sarima_model, open('/app/backend/ml_models/sarima_model.pkl', 'wb'))
    print(f"\n✓ Saved SARIMA model")
    
    # Step 4: Train LSTM
    lstm_model, lstm_scaler, lstm_metrics = train_lstm_model(df)
    
    # Save LSTM model and scaler
    lstm_model.save('/app/backend/ml_models/lstm_model.h5')
    pickle.dump(lstm_scaler, open('/app/backend/ml_models/lstm_scaler.pkl', 'wb'))
    print(f"\n✓ Saved LSTM model and scaler")
    
    # Step 5: Compare models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    models = ['SARIMA', 'LSTM']
    metrics = [sarima_metrics, lstm_metrics]
    
    comparison_df = pd.DataFrame(metrics, index=models)
    print("\n", comparison_df)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(['mae', 'rmse', 'mape']):
        values = [m[metric] for m in metrics]
        axes[i].bar(models, values, color=['skyblue', 'lightgreen'])
        axes[i].set_title(metric.upper())
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for j, v in enumerate(values):
            axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/app/ml_training/model_comparison_forecast.png', dpi=100)
    plt.close()
    print(f"\n✓ Saved model comparison")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\nOutput Files:")
    print("  - ticket_history.csv (731 days of data)")
    print("  - sarima_model.pkl")
    print("  - lstm_model.h5")
    print("  - lstm_scaler.pkl")
    print("  - Visualizations in /app/ml_training/")

if __name__ == "__main__":
    main()
