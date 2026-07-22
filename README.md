# DISoptimizer

DISoptimizer is a go-to Python toolkit for smarter water disinfection. Simply provide your raw water conditions, and our tool will deliver key predictions:

-   **Optimal Chlorine Dosage**: The precise amount needed for effective disinfection.
    
-   **Residual Chlorine**: The expected chlorine level after a specified contact time.
    
-   **THM Formation**: The predicted concentration of Trihalomethane byproducts.
    

Make data-driven decisions for safer, compliant drinking water.

# Content

- **Overview**
- **System requirements**
- **Installation Guide**
- **Demo**
- **Instructions for use**
- **Reproduction Instructions**


## Overview

 
Drinking water disinfection has been highly effective in preventing waterborne disease; however, the formation of disinfection byproducts poses a significant public health concern. Mitigating DBP formation without compromising disinfection efficiency remains a difficult yet essential challenge. DISoptimizer is a Python-based, machine learning-driven framework designed to address this challenge. It intelligently predicts the **optimal chlorine dosage**, the resulting **residual chlorine**, and **THM formation** based on given water quality parameters. The package provides a user-friendly interface for water utilities to make data-driven decisions, aiming to enhance both public health protection and regulatory compliance.


## System requirements

**1.** Software Dependencies & Operating Systems

DISoptimizer  has been successfully tested and run on the following software environments:

    **Programming Language:**
    *   Python (3.11)
       
    **Python Packages:**
    *   numpy (1.25.2)
    *   pandas (2.2.2)
    *   scikit-learn (1.5.1)
    *   XGBoost (1.7.6)
    *   LightGBM (4.6.0)
    *   CatBoost (1.2.8)
    *   Bayesian Optimization (3.0.0)
 
    **Operating Systems:**
    *   macOS (Monterey 12.6)

**2.** Non-Standard Hardware

No special or non-standard hardware is required. The software runs efficiently on a standard desktop or laptop computer.


## Installation guide

This guide will help you set up the DISoptimizer environment.

1.   Python 3.11 or higher 
2.   Create a virtual environment (optional)
         
            python -m venv disoptimizer_env
            source disoptimizer_env/bin/activate

3.  Download the source code
	
	       git clone https://github.com/pinfish94/DISoptimizer.git
           cd DISoptimizer

4.   Python packages:
  
            pip install -r requirements.txt
 
 #Typical Install Time

On a standard desktop computer with a standard internet connection, the entire installation process (after downloading the code) typically takes **5 minutes**.

## Demo
This demo illustrates how to:

1.  Train the chlorine consumption prediction model
    
2.  Train the THM prediction model
    
3.  Run the disinfection optimization module

- Step 1   Train the chlorine consumption prediction model

       python “predict_Cl2consumption.py”

   Expected output
 
       train R²:    | RMSE:  
       test R²:     | RMSE:
      
       
- Step 2   Train the THM Prediction Model

       python “predict_THM.py”

   Expected output
 
       train R²:    | RMSE:  
       test  R²:    | RMSE:

 
 - Step 3   Run the Optimization
 **This step uses the pre-trained models to find the optimal chlorine dosage for a new set of water quality data.
    
        python "optimizer_csafe.py"

**Expected Run Time**

The execution time varies for different steps of DISoptimizer:

- **Training Models** (`predict_Cl2consumption.py` and `predict_THM.py`): 
  Typically completes in **about 1-2 minutes** on a normal desktop computer.

- **Running Optimization** (`optimizer_csafe.py`):
  The optimization process takes approximately **5-30 seconds per water sample**. The total time depends on the number of samples in your input file - more samples will require proportionally longer processing time.


## Instructions for use

#### 1. Prepare Your Data
Prepare your data according to the format in the **"DISoptimizer"** sheet of `Dataset.xlsx`


### 2. Configure the Script
Open `optimuzer_csafe.py` and modify the following lines in the `if __name__ == "__main__":` section:

    if __name__ == "__main__":
         input_file = "your_data.xlsx"  # Change to your Excel file name
         sheetname = "DISoptimizer"     # Change if using different sheet name
        
### 3. Set parameters
Open `optimuzer_csafe.py` and modify the following lines in 

`HARD_L, HARD_U = 0.3, 1.5 
 DOSE_MIN, DOSE_MAX = 0.1, 10.0  
 CSAFE = 0.80 A_WEIGHT = 0.3 ` # constraints and preference settings 

### 4. Run the Optimization
   


## License

The software is released under the MIT License. Parts of the underlying disinfection optimization method are covered by a pending patent application, and the open-source license applies to the code only.
