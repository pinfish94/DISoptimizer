# DISoptimizer

DISoptimizer is a go-to Python toolkit for smarter water disinfection. Simply provide your raw water conditions, and our tool will deliver key predictions:

-   **Optimal Chlorine Dosage**: The precise amount needed for effective disinfection.
    
-   **Residual Chlorine**: The expected chlorine level after a specified contact time.
    
-   **THM Formation**: The predicted concentration of Trihalomethane byproducts.
    

Make data-driven decisions for safer, compliant drinking water.

# Content

-   **Overview**
-  **System requirements**
-  **Installation Guide**
- **Demo**
- **Instructions for use**


## Overview

 
Drinking water disinfection has been highly effective in preventing waterborne disease; however, the formation of disinfection byproducts poses a significant public health concern. Mitigating DBP formation without compromising disinfection efficiency remains a difficult yet essential challenge. DISoptimizer is a Python-based, machine learning-driven framework designed to address this challenge. It intelligently predicts the **optimal chlorine dosage**, the resulting **residual chlorine**, and **THM formation** based on given water quality parameters. The package provides a user-friendly interface for water utilities to make data-driven decisions, aiming to enhance both public health protection and regulatory compliance.


## System requirements

**1.** Software Dependencies & Operating Systems

DISoptimizer requires the following software environment. The specified versions are those with which the framework has been verified.

   **Programming Language:**
       Python (3.11)
       
  **Core Python Packages:**
    numpy (>= 1.21.0)
    pandas (>= 1.3.0)
    scikit-learn (>= 1.0.0)

  **Operating Systems:**
    macOS (Monterey 12.6)

**2.** Versions Tested On

The software has been successfully tested and run on the following specific environments:

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
