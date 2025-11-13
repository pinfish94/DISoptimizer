from scipy.optimize import minimize_scalar
import numpy as np
import pandas as pd
from predict_thm import predict_thm
from predict_Cl2consumption import predict_Cl2_consumpution

lambda_penalty = 300

def batch_optimize(input_excel_path, output_excel_path,sheet_name=None):

    try:
        input_data = pd.read_excel(input_excel_path,sheet_name=sheet_name)
        print(f"read {len(input_data)} water quality")

        required_cols = ['pH', 'UV254', 'DOC', 'TN', 'SUVA', 'Br', 'EEM_I', 'EEM_V']
        missing_cols = set(required_cols) - set(input_data.columns)
        if missing_cols:
            raise ValueError(f"lack: {missing_cols}")

        results = []

        for idx, row in input_data.iterrows():
            print(f"\n {idx + 1}/{len(input_data)} ...")

            X_wo_dose = {col: float(row[col]) for col in required_cols}

            best_solution = main_optimization(X_wo_dose)

            if best_solution:
                result = {
                    'ID': idx + 1,
                    'pH': X_wo_dose['pH'],
                    'UV254': X_wo_dose['UV254'],
                    'DOC': X_wo_dose['DOC'],
                    'optimal chlorine dose': best_solution['cl_dose'],
                    'predicted chlorine consumption': best_solution['cl_consumed'],
                    'predicted chlorine residual': best_solution['free_cl'],
                    'predicted THM formation': best_solution['thm_pred']
                }
                results.append(result)

        results_df = pd.DataFrame(results)
        results_df.to_excel(output_excel_path, index=False)
        print(f"\nresult save : {output_excel_path}")

        return results_df

    except Exception as e:
        print(f"error: {str(e)}")
        return None


def main_optimization(X_wo_dose):
    try:
        candidates = global_feasible_search(X_wo_dose)
        final_results = []

        for candidate in candidates:
            refined = local_refinement(candidate['cl_dose'], X_wo_dose)
            if refined and refined['is_feasible']:
                final_results.append(refined)

        if not final_results:
            raise RuntimeError("error")

        best_solution = min(final_results, key=lambda x: x['objective_value'])
        return best_solution

    except Exception as e:
        print(f"error: {str(e)}")
        return None


def global_feasible_search(X_wo_dose):
    doses = np.exp(np.linspace(np.log(0.5), np.log(8.0), 160))
    feasible_solutions = []

    for dose in doses:
        result = evaluate_dose(dose, X_wo_dose)
        if result and result['is_feasible']:
            feasible_solutions.append(result)

    if not feasible_solutions:
        raise ValueError("error")

    feasible_solutions.sort(key=lambda x: x['objective_value'])
    return feasible_solutions[:3]


def local_refinement(initial_dose, X_wo_dose):
    def objective_func(cl_dose):
        result = evaluate_dose(cl_dose, X_wo_dose)
        if not result or not result['is_feasible']:
            return 1e10
        return result['objective_value']

    bounds = (max(0.5, initial_dose - 0.5), min(8.0, initial_dose + 0.5))
    res = minimize_scalar(objective_func, bounds=bounds, method='bounded',
                          options={'xatol': 0.001, 'maxiter': 50})
    return evaluate_dose(res.x, X_wo_dose)


def evaluate_dose(cl_dose, X_wo_dose):

    try:
        cl_used = predict_Cl2_consumpution(make_input_for_cl2(cl_dose, X_wo_dose))
        if isinstance(cl_used, (np.ndarray, pd.Series)):
            cl_used = float(cl_used[0])

        free_cl = cl_dose - cl_used
        thm_pred = predict_thm(make_input_for_thm(cl_dose, cl_used, X_wo_dose))
        if isinstance(thm_pred, (np.ndarray, pd.Series)):
            thm_pred = float(thm_pred[0])


        if free_cl < 0.5 or free_cl > 1.5:
            return {
                'cl_dose': cl_dose,
                'cl_consumed': cl_used,
                'free_cl': free_cl,
                'thm_pred': thm_pred,
                'is_feasible': False,
                'objective_value': 1e10
            }


        if free_cl <= 1.0:
            penalty = 0
        else:
            penalty = (free_cl - 1.0) ** 2

        objective_value = thm_pred + lambda_penalty * penalty

        return {
            'cl_dose': cl_dose,
            'cl_consumed': cl_used,
            'free_cl': free_cl,
            'thm_pred': thm_pred,
            'is_feasible': True,
            'objective_value': objective_value
        }

    except Exception as e:
        print(f" {cl_dose} error: {str(e)}")
        return None


def make_input_for_cl2(cl_dose, X_wo_dose):
    input_data = {**X_wo_dose, 'chlorine dose': cl_dose}
    expected_order = ['pH', 'UV254', 'DOC', 'TN', 'SUVA', 'Br',
                      'chlorine dose', 'EEM_I', 'EEM_V']
    return pd.DataFrame([input_data], columns=expected_order)


def make_input_for_thm(cl_dose, cl_consumed, X_wo_dose):
    input_data = {**X_wo_dose,
                  'chlorine dose': cl_dose,
                  'chlorine comsumption': cl_consumed}
    expected_order = ['pH', 'UV254', 'DOC', 'TN', 'SUVA', 'Br',
                      'chlorine dose', 'chlorine comsumption', 'EEM_I', 'EEM_V']
    return pd.DataFrame([input_data], columns=expected_order)


if __name__ == "__main__":
    input_file = "Dataset.xlsx"
    sheetname = "DISoptimizer"
    output_file = "cl2dose.xlsx"

    results = batch_optimize(input_file, output_file,sheet_name=sheetname)
    if results is not None:
        print("\n result:")
        print(results.head())
