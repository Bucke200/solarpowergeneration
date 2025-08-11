# Solar Power Forecasting Model

A comprehensive machine learning project for predicting hourly solar power generation using weather data and temporal features. This project demonstrates a complete end-to-end ML pipeline from synthetic data generation to model evaluation and future forecasting.

## 🌞 Project Overview

This research project, developed as part of the **1M1B (One Million for One Billion) internship program**, addresses the critical challenge of renewable energy forecasting. The model predicts solar power output based on weather conditions and temporal patterns, which is essential for:

- Grid stability and energy planning
- Solar farm operations optimization
- Energy trading and pricing strategies
- Battery storage management
- Smart grid integration

## 📊 Model Output & Results

### Model Performance Visualization

![Model Performance](<public/Screenshot%20(800).png>)
_Actual vs Predicted Solar Power Output showing model accuracy on test data_

### Future Forecast Dashboard

![Future Forecast Dashboard](public/Screenshot%202025-08-11%20110915.png)
_Comprehensive 7-day forecast dashboard showing power predictions, daily energy production, weather correlations, and hourly patterns_

## 🚀 Features

- **Complete ML Pipeline**: End-to-end workflow from data generation to visualization
- **Synthetic Data Generation**: Creates realistic solar power and weather datasets
- **Advanced Feature Engineering**: Temporal and meteorological feature extraction
- **XGBoost Regression**: State-of-the-art gradient boosting for prediction
- **Future Forecasting**: 7-day ahead solar power predictions
- **Comprehensive Visualization**: Multiple charts for model performance analysis
- **Performance Metrics**: MAE, RMSE evaluation with detailed reporting

## 📁 Project Structure

```
solar-power-forecasting/
├── solar_forecasting_model.py    # Main ML pipeline script
├── future_prediction.py          # Future forecasting module
├── power_generation.csv          # Generated solar power data
├── weather_data.csv              # Generated weather data
├── future_solar_predictions.csv  # Future predictions output
├── daily_forecast_summary.csv    # Daily forecast summary
├── README.md                     # Project documentation
└── .kiro/specs/                  # Project specifications
    ├── requirements.md
    ├── design.md
    └── tasks.md
```

## 🛠️ Technology Stack

- **Python 3.8+**
- **XGBoost**: Gradient boosting framework
- **Scikit-learn**: Machine learning utilities
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Datetime**: Time series handling

## 📊 Model Performance

The trained XGBoost model achieves:

- **Mean Absolute Error (MAE)**: ~2.6 MW
- **Root Mean Square Error (RMSE)**: ~4.4 MW
- **Training Data**: 800 samples (80%)
- **Test Data**: 200 samples (20%)
- **Features**: 6 (weather + temporal)

## 🔧 Installation & Setup

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd solar-power-forecasting
   ```

2. **Install required packages**:

   ```bash
   pip install xgboost scikit-learn matplotlib pandas numpy
   ```

3. **Run the main model**:

   ```bash
   python solar_forecasting_model.py
   ```

4. **Generate future predictions**:
   ```bash
   python future_prediction.py
   ```

## 📈 Usage

### Basic Model Training and Evaluation

```python
# Run the complete ML pipeline
python solar_forecasting_model.py
```

This script will:

1. Generate synthetic solar power and weather data (1000 hours)
2. Preprocess and merge datasets with feature engineering
3. Train XGBoost regression model
4. Evaluate performance with MAE/RMSE metrics
5. Create visualization comparing actual vs predicted values

### Future Forecasting

```python
# Generate 7-day solar power forecast
python future_prediction.py
```

This module will:

1. Load the trained model
2. Generate future weather forecast data
3. Make solar power predictions for the next 7 days
4. Create comprehensive forecast visualizations
5. Export predictions and daily summaries

## 📊 Data Schema

### Power Generation Data

- `datetime`: Hourly timestamps
- `power_output_mw`: Solar power output (0-100 MW)

### Weather Data

- `datetime`: Hourly timestamps
- `temperature_celsius`: Ambient temperature (10-35°C)
- `cloud_cover_percent`: Cloud coverage (0-100%)
- `ghi`: Global Horizontal Irradiance (0-1000 W/m²)

### Engineered Features

- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0-6)
- `month`: Month (1-12)

## 🎯 Key Insights

### Model Findings

- Solar power follows predictable diurnal patterns (zero at night, peak at noon)
- Weather conditions (cloud cover, solar irradiance) significantly impact output
- Temporal features capture seasonal and daily cycles effectively
- XGBoost successfully learns complex weather-power relationships

### Forecast Analysis

- Best production typically occurs during midday hours (11 AM - 2 PM)
- Cloud cover shows strong inverse correlation with power output
- Daily energy production varies based on weather conditions
- Model provides reliable short-term forecasting capabilities

## 📋 Future Enhancements

- [ ] Integration with real weather API data
- [ ] Advanced time series forecasting (LSTM, Prophet)
- [ ] Seasonal pattern analysis and modeling
- [ ] Multi-location solar farm predictions
- [ ] Real-time model updating and retraining
- [ ] Uncertainty quantification in predictions
- [ ] Integration with energy storage optimization

## 🔬 Research Context

This project was developed as part of the **1M1B (One Million for One Billion)** internship program, focusing on technology solutions that can benefit billions globally. The research contributes to:

- Renewable energy integration challenges
- Smart grid planning and management
- Sustainable energy transition in emerging markets
- ML-driven approaches for clean energy optimization

## 📝 License

This project is developed for research and educational purposes as part of the 1M1B internship program.

## 👥 Contributing

This is a research project developed during an internship. For questions or collaboration opportunities, please reach out through the 1M1B program channels.

## 📞 Contact

Developed as part of the 1M1B internship program.
For inquiries related to this research project, please contact through official program channels.

---

**Note**: This project uses synthetic data for demonstration purposes. For production deployment, integrate with real weather forecast APIs and actual solar power generation data.
