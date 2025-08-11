"""
Solar Power Forecasting Model
============================

A complete machine learning pipeline for predicting hourly solar power generation
using weather data. This script demonstrates the entire data science workflow from
synthetic data generation to model evaluation and visualization.

Author: AI Assistant
Date: 2025-01-08
"""

# Import required libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("Solar Power Forecasting Model")
print("=" * 40)
print("Starting the complete ML pipeline...\n")

# ============================================================================
# STEP 1: SETUP AND SAMPLE DATA GENERATION
# ============================================================================

def generate_power_data():
    """
    Generate realistic synthetic solar power generation data.
    
    Returns:
        pd.DataFrame: DataFrame with datetime and power_output_mw columns
    """
    print("Generating power generation data...")
    
    # Create 1000 hours of data (approximately 42 days)
    start_date = datetime(2024, 6, 1)  # Start in summer for better solar patterns
    dates = [start_date + timedelta(hours=i) for i in range(1000)]
    
    power_output = []
    
    for date in dates:
        hour = date.hour
        
        # Solar power is zero during nighttime (6 PM to 6 AM)
        if hour < 6 or hour >= 18:
            power = 0.0
        else:
            # Create realistic diurnal pattern using sine function
            # Peak power around noon (hour 12)
            hour_angle = (hour - 6) * np.pi / 12  # 0 to π for daylight hours
            base_power = 80 * np.sin(hour_angle)  # Max 80 MW
            
            # Add some realistic variability (weather effects)
            noise = np.random.normal(0, 5)  # ±5 MW noise
            power = max(0, base_power + noise)  # Ensure non-negative
        
        power_output.append(power)
    
    # Create DataFrame
    power_df = pd.DataFrame({
        'datetime': dates,
        'power_output_mw': power_output
    })
    
    # Save to CSV
    power_df.to_csv('power_generation.csv', index=False)
    print(f"✓ Generated {len(power_df)} rows of power data")
    print(f"✓ Power range: {power_df['power_output_mw'].min():.1f} - {power_df['power_output_mw'].max():.1f} MW")
    
    return power_df

# Generate power generation data
power_data = generate_power_data()

def generate_weather_data():
    """
    Generate realistic synthetic weather data with proper correlations.
    
    Returns:
        pd.DataFrame: DataFrame with datetime, temperature_celsius, cloud_cover_percent, and ghi columns
    """
    print("\nGenerating weather data...")
    
    # Use same datetime range as power data
    start_date = datetime(2024, 6, 1)
    dates = [start_date + timedelta(hours=i) for i in range(1000)]
    
    temperature = []
    cloud_cover = []
    ghi = []
    
    for date in dates:
        hour = date.hour
        
        # Temperature follows daily pattern (cooler at night, warmer during day)
        temp_base = 22 + 8 * np.sin((hour - 6) * np.pi / 12)  # 14-30°C base range
        temp_noise = np.random.normal(0, 2)  # ±2°C variability
        temp = max(10, min(35, temp_base + temp_noise))  # Clamp to 10-35°C
        temperature.append(temp)
        
        # Cloud cover (0-100%) - random but affects GHI
        cloud = np.random.uniform(0, 100)
        cloud_cover.append(cloud)
        
        # GHI (Global Horizontal Irradiance) - correlated with time and inverse to clouds
        if hour < 6 or hour >= 18:
            # No solar irradiance at night
            irradiance = 0.0
        else:
            # Base irradiance follows sun angle
            hour_angle = (hour - 6) * np.pi / 12
            base_ghi = 800 * np.sin(hour_angle)  # Max 800 W/m²
            
            # Reduce GHI based on cloud cover (inverse relationship)
            cloud_factor = 1 - (cloud / 100) * 0.7  # Clouds reduce GHI by up to 70%
            irradiance = max(0, base_ghi * cloud_factor)
            
            # Add some noise
            irradiance += np.random.normal(0, 20)
            irradiance = max(0, irradiance)
        
        ghi.append(irradiance)
    
    # Create DataFrame
    weather_df = pd.DataFrame({
        'datetime': dates,
        'temperature_celsius': temperature,
        'cloud_cover_percent': cloud_cover,
        'ghi': ghi
    })
    
    # Save to CSV
    weather_df.to_csv('weather_data.csv', index=False)
    print(f"✓ Generated {len(weather_df)} rows of weather data")
    print(f"✓ Temperature range: {weather_df['temperature_celsius'].min():.1f} - {weather_df['temperature_celsius'].max():.1f} °C")
    print(f"✓ GHI range: {weather_df['ghi'].min():.1f} - {weather_df['ghi'].max():.1f} W/m²")
    
    return weather_df

# Generate weather data
weather_data = generate_weather_data()

# ============================================================================
# STEP 2: DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*50)
print("STEP 2: DATA PREPROCESSING AND FEATURE ENGINEERING")
print("="*50)

# Load the generated CSV files
print("Loading and merging datasets...")
power_df = pd.read_csv('power_generation.csv')
weather_df = pd.read_csv('weather_data.csv')

# Convert datetime columns to proper datetime objects
power_df['datetime'] = pd.to_datetime(power_df['datetime'])
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

# Merge the DataFrames on datetime
merged_df = pd.merge(power_df, weather_df, on='datetime', how='inner')
print(f"✓ Successfully merged datasets: {len(merged_df)} rows")
print(f"✓ Columns: {list(merged_df.columns)}")

# Check for and handle missing values
print(f"\nChecking for missing values...")
missing_values = merged_df.isnull().sum()
print(f"Missing values per column:\n{missing_values}")

if missing_values.sum() > 0:
    print("Handling missing values with forward fill...")
    merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
    print("✓ Missing values handled")
else:
    print("✓ No missing values found")

# Create temporal features from datetime
print(f"\nCreating temporal features...")
merged_df['hour'] = merged_df['datetime'].dt.hour
merged_df['day_of_week'] = merged_df['datetime'].dt.dayofweek
merged_df['month'] = merged_df['datetime'].dt.month

print(f"✓ Created temporal features: hour, day_of_week, month")
print(f"✓ Final dataset shape: {merged_df.shape}")

# Validate feature set
expected_features = ['temperature_celsius', 'cloud_cover_percent', 'ghi', 'hour', 'day_of_week', 'month']
available_features = [col for col in expected_features if col in merged_df.columns]
print(f"✓ Available features: {available_features}")
print(f"✓ Target variable: power_output_mw")

# ============================================================================
# STEP 3: MODEL TRAINING
# ============================================================================

print("\n" + "="*50)
print("STEP 3: MODEL TRAINING")
print("="*50)

# Define feature matrix (X) and target variable (y)
feature_columns = ['temperature_celsius', 'cloud_cover_percent', 'ghi', 'hour', 'day_of_week', 'month']
X = merged_df[feature_columns]
y = merged_df['power_output_mw']

print(f"Feature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")
print(f"Features: {feature_columns}")

# Split data chronologically (80% train, 20% test) - NO SHUFFLING
split_index = int(0.8 * len(merged_df))
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"\nData split (chronological order):")
print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")
print(f"✓ Train period: {merged_df['datetime'].iloc[0]} to {merged_df['datetime'].iloc[split_index-1]}")
print(f"✓ Test period: {merged_df['datetime'].iloc[split_index]} to {merged_df['datetime'].iloc[-1]}")

# Initialize and train XGBoost model
print(f"\nTraining XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(X_train, y_train)
print(f"✓ Model training completed successfully")
print(f"✓ Model type: {type(model).__name__}")
print(f"✓ Number of estimators: {model.n_estimators}")
print(f"✓ Learning rate: {model.learning_rate}")

# ============================================================================
# STEP 4: MODEL EVALUATION
# ============================================================================

print("\n" + "="*50)
print("STEP 4: MODEL EVALUATION")
print("="*50)

# Generate predictions on test set
print("Generating predictions on test set...")
y_pred = model.predict(X_test)
print(f"✓ Generated {len(y_pred)} predictions")

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print metrics with clear formatting
print(f"\nModel Performance Metrics:")
print(f"=" * 30)
print(f"Mean Absolute Error (MAE):  {mae:.2f} MW")
print(f"Root Mean Square Error (RMSE): {rmse:.2f} MW")
print(f"=" * 30)

# Additional context
print(f"\nActual power output range in test set: {y_test.min():.1f} - {y_test.max():.1f} MW")
print(f"Predicted power output range: {y_pred.min():.1f} - {y_pred.max():.1f} MW")
print(f"Mean actual power output: {y_test.mean():.1f} MW")
print(f"Mean predicted power output: {y_pred.mean():.1f} MW")

# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================

print("\n" + "="*50)
print("STEP 5: VISUALIZATION")
print("="*50)

# Create performance visualization
print("Creating visualization plot...")

# Get test set datetime values for x-axis
test_dates = merged_df['datetime'].iloc[split_index:].values

# Create figure and axis
plt.figure(figsize=(15, 8))

# Plot actual vs predicted values
plt.plot(test_dates, y_test.values, label='Actual Power Output', color='blue', linewidth=1.5, alpha=0.8)
plt.plot(test_dates, y_pred, label='Predicted Power Output', color='red', linewidth=1.5, alpha=0.8)

# Format the plot
plt.title('Actual vs. Predicted Solar Power Output', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Power Output (MW)', fontsize=12)
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Add performance metrics as text on the plot
textstr = f'MAE: {mae:.2f} MW\nRMSE: {rmse:.2f} MW'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# Display the plot
print("✓ Visualization created successfully")
print("✓ Displaying plot...")
plt.show()

print(f"\n" + "="*50)
print("SOLAR POWER FORECASTING MODEL COMPLETE")
print("="*50)
print(f"✓ Generated synthetic datasets (1000 hours each)")
print(f"✓ Preprocessed and merged data with temporal features")
print(f"✓ Trained XGBoost model (n_estimators=1000, lr=0.05)")
print(f"✓ Evaluated model performance (MAE: {mae:.2f} MW, RMSE: {rmse:.2f} MW)")
print(f"✓ Created visualization comparing actual vs predicted values")
print(f"\nFiles created:")
print(f"  - power_generation.csv")
print(f"  - weather_data.csv")
print(f"  - solar_forecasting_model.py (this script)")
print("="*50)

# Final validation - ensure script executed successfully
if __name__ == "__main__":
    print(f"\n✅ Script execution completed successfully!")
    print(f"   All 5 steps completed in sequential order:")
    print(f"   1. ✓ Data generation (power_generation.csv, weather_data.csv)")
    print(f"   2. ✓ Data preprocessing and feature engineering")
    print(f"   3. ✓ Model training (XGBoost)")
    print(f"   4. ✓ Model evaluation (MAE, RMSE)")
    print(f"   5. ✓ Visualization (Actual vs Predicted plot)")
    print(f"\n🎯 Ready for analysis and further development!")