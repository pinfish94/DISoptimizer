# DISoptimizer

DISoptimizer is a go-to Python toolkit for smarter water disinfection. Simply provide your raw water conditions, and our tool will deliver key predictions:

-   **Optimal Chlorine Dosage**: The precise amount needed for effective disinfection.
    
-   **Residual Chlorine**: The expected chlorine level after a specified contact time.
    
-   **THM4 Formation**: The predicted concentration of Trihalomethane byproducts.
    

Make data-driven decisions for safer, compliant drinking water.

# Content

-   **Overview**
-  **System requirements**
-  **Installation Guide**
-  **Demo**
-  **Instructions for use**

## Quick Start

### 1. Train Prediction Models
First, train and save the required models:

# Train and save chlorine consumption model
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
