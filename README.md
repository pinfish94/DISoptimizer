# DISoptimizer
Paper code availability 

# DISoptimizer: Water Disinfection Optimization

A machine learning framework for optimizing chlorine dose in drinking water treatment to minimize THM formation without compromising disinfection

## Quick Start

### 1. Train Prediction Models
First, train and save the required models:

# Train and save Cl2 consumption model
from train_cl2_consumption import train_and_save_model
train_and_save_model()

# Train and save THM prediction model  
from train_thm import train_and_save_model
train_and_save_model()

### 2. Run Optimization
from predict_Cl2_dose import batch_optimize

results = batch_optimize(
    input_excel_path="Dataset.xlsx",
    output_excel_path="optimized_doses.xlsx",
    sheet_name="DISoptimizer"
)

### Input Format

Required columns: pH, UV254, DOC, TN, SUVA, Br, EEM_I, EEM_V

### Output

Optimal chlorine dose (mg/L)
Predicted chlorine consumption
Free chlorine residual (mg/L)
THM concentration (μg/L)
