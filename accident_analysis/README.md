# Accident Analysis Dashboard

A complete solution for analyzing accident data, identifying high-risk locations, and predicting future accident severity.

## Project Structure
```
accident_analysis/
├── data/                    # Raw and processed data files
│   └── accident_data.csv    # Sample accident dataset
├── powerbi/                 # Power BI resources
│   ├── AccidentAnalysis_Template.pbix  # Dashboard template
│   └── powerbi_script.txt   # DAX measures and Power Query M code
└── python_scripts/          # Predictive analytics scripts
    ├── predictive_model.py  # Machine learning model
    └── requirements.txt     # Python dependencies
```

## Setup Instructions

### 1. Python Environment Setup
```bash
cd accident_analysis/python_scripts
pip install -r requirements.txt
```

### 2. Running Predictive Model
```bash
python predictive_model.py
```
This will:
- Train a Random Forest model on the accident data
- Save the model as `accident_severity_model.pkl`
- Print model accuracy metrics

### 3. Power BI Integration
1. Open `AccidentAnalysis_Template.pbix`
2. In Power Query Editor:
   - Replace the sample data path with your actual data path
   - Paste the M code from `powerbi_script.txt`
3. In Report View:
   - Add the DAX measures from `powerbi_script.txt`
   - Configure visualizations as needed

### 4. Connecting Python to Power BI
1. Enable Python scripting in Power BI (Options → Python scripting)
2. Use this code to load predictions:
```python
import pandas as pd
import joblib

# Load trained model
model = joblib.load('accident_severity_model.pkl')

# Make predictions on current dataset
dataset['PredictedSeverity'] = model.predict(dataset[features])
```

## Key Features
- **Hotspot Identification**: Geographic visualization of high-risk locations
- **Trend Analysis**: Time-series patterns and seasonal variations
- **Predictive Analytics**: Severity forecasting using machine learning
- **Interactive Filters**: Drill-down capability by location, cause, and time

## Next Steps
- [ ] Import your real accident dataset
- [ ] Customize the Power BI dashboard
- [ ] Schedule regular data refreshes
- [ ] Deploy to Power BI Service for sharing