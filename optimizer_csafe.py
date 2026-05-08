# -*- coding: utf-8 -*-
"""
lambda_crit by marginal-area equality and user weight a for optimal chlorine dose.

Main idea:
- Given hard residual chlorine boundaries Cmin (= HARD_L) and Csafe.
- For each water-quality condition X, lambda_crit(X) is computed deterministically:

    integral from Cmin to Csafe of T(C) dC
    =
    lambda_crit * integral from Cmin to Csafe of P(C) dC

  where:

    T(C) = (THM(C) - THM_min) / THM_max
    P(C) = ((C - Csafe) / Csafe)^2

- Given user preference a in [0, 1], the objective function is:

    J = (1 - a) * THM / THM_max
        + a * lambda_crit * ((C - Csafe) / Csafe)^2

- The optimal chlorine dose is searched within the feasible dose domain.
- HARD_L and HARD_U are used as hard residual chlorine constraints.

Dependencies:
    pip install numpy pandas scipy openpyxl

Make sure predict_thm.py and predict_Cl2consumption.py are in the same directory.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

try:
    from predict_thm import predict_thm
    from predict_Cl2consumption import predict_Cl2_consumpution
    print("Prediction models imported successfully.")
except ImportError as e:
    print("Error: Failed to import predict_thm.py or predict_Cl2consumption.py:", e)
    raise


HARD_L, HARD_U = 0.3, 1.5

DOSE_MIN, DOSE_MAX = 0.1, 10.0

CSAFE = 0.80
A_WEIGHT = 0.5

N_COARSE = 400
N_REFINE = 300

TOPK_CAND = 50
XATOL = 1e-5
MAXITER = 60

C_WINDOW_FOR_LAM = 1e-9
MIN_P_AREA = 1e-12
DIAG_PRINT = True


def _to_float(x) -> float:
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    if isinstance(x, np.ndarray):
        return float(x.ravel()[0])
    if isinstance(x, list):
        return float(x[0])
    return float(x)


def make_input_for_cl2(cl_dose: float, X_wo_dose: dict) -> pd.DataFrame:
    input_data = {
        "pH": X_wo_dose["pH"],
        "UV254": X_wo_dose["UV254"],
        "DOC": X_wo_dose["DOC"],
        "TN": X_wo_dose["TN"],
        "Br": X_wo_dose["Br"],
        "EEM_I": X_wo_dose["EEM_I"],
        "EEM_V": X_wo_dose["EEM_V"],
        "Chlorine dose": float(cl_dose),
    }
    cols = ["pH", "UV254", "DOC", "TN", "Br", "EEM_I", "EEM_V", "Chlorine dose"]
    return pd.DataFrame([input_data], columns=cols)


def make_input_for_thm(cl_dose: float, cl_consumed: float, X_wo_dose: dict) -> pd.DataFrame:
    input_data = {
        "pH": X_wo_dose["pH"],
        "UV254": X_wo_dose["UV254"],
        "DOC": X_wo_dose["DOC"],
        "TN": X_wo_dose["TN"],
        "Br": X_wo_dose["Br"],
        "EEM_II": X_wo_dose["EEM_II"],
        "Chlorine dose": float(cl_dose),
        "Chlorine consumption": float(cl_consumed),
    }
    cols = [
        "pH",
        "UV254",
        "DOC",
        "TN",
        "Br",
        "EEM_II",
        "Chlorine dose",
        "Chlorine consumption",
    ]
    return pd.DataFrame([input_data], columns=cols)


def evaluate_dose_raw(dose: float, X_wo_dose: dict):
    try:
        df_cl = make_input_for_cl2(dose, X_wo_dose)
        cl_consumed = _to_float(predict_Cl2_consumpution(df_cl))
        free_cl = float(dose) - float(cl_consumed)

        if (not np.isfinite(free_cl)) or (free_cl < HARD_L) or (free_cl > HARD_U):
            return None

        df_thm = make_input_for_thm(dose, cl_consumed, X_wo_dose)
        thm_pred = _to_float(predict_thm(df_thm))

        if not np.isfinite(thm_pred):
            return None

        return {
            "cl_dose": float(dose),
            "cl_consumed": float(cl_consumed),
            "free_cl": float(free_cl),
            "thm_pred": float(thm_pred),
        }

    except Exception:
        return None


def build_point_cloud(X_wo_dose: dict, Csafe: float):
    points = []

    doses = np.linspace(DOSE_MIN, DOSE_MAX, N_COARSE)
    for d in doses:
        r = evaluate_dose_raw(d, X_wo_dose)
        if r:
            points.append(r)

    if not points:
        return []

    near = [p for p in points if (Csafe - 0.20) <= p["free_cl"] <= (Csafe + 0.20)]
    if near:
        dmin = max(DOSE_MIN, min(p["cl_dose"] for p in near) - 0.5)
        dmax = min(DOSE_MAX, max(p["cl_dose"] for p in near) + 0.5)
        doses_fine = np.linspace(dmin, dmax, N_REFINE)

        for d in doses_fine:
            r = evaluate_dose_raw(d, X_wo_dose)
            if r:
                points.append(r)

    uniq = {}
    for p in points:
        key = round(p["cl_dose"], 6)
        if key not in uniq:
            uniq[key] = p

    return list(uniq.values())


def trapezoid_integral(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def compute_lambda_crit_marginal_area(points, Cmin: float, Csafe: float):
    if (not points) or (Csafe <= Cmin):
        return 0.0, {
            "n_used": 0,
            "A_thm": np.nan,
            "A_pen": np.nan,
            "THMmin": np.nan,
            "THMmax": np.nan,
        }

    band = [
        p for p in points
        if (Cmin <= p["free_cl"] <= Csafe) and np.isfinite(p["thm_pred"])
    ]

    if len(band) < 3:
        return 0.0, {
            "n_used": len(band),
            "A_thm": np.nan,
            "A_pen": np.nan,
            "THMmin": np.nan,
            "THMmax": np.nan,
        }

    band.sort(key=lambda z: z["free_cl"])

    C = np.array([p["free_cl"] for p in band], dtype=float)
    THM = np.array([p["thm_pred"] for p in band], dtype=float)

    THMmin = float(np.min(THM))
    THMmax = float(np.max(THM))

    denom = max(THMmax, C_WINDOW_FOR_LAM)
    T = (THM - THMmin) / denom

    P = ((C - Csafe) / max(Csafe, C_WINDOW_FOR_LAM)) ** 2

    A_thm = trapezoid_integral(C, T)
    A_pen = trapezoid_integral(C, P)

    if (not np.isfinite(A_thm)) or (not np.isfinite(A_pen)) or (A_pen <= MIN_P_AREA):
        lam = 0.0
    else:
        lam = float(A_thm / A_pen)
        if lam < 0:
            lam = 0.0

    diag = {
        "n_used": int(len(band)),
        "A_thm": float(A_thm),
        "A_pen": float(A_pen),
        "THMmin": float(THMmin),
        "THMmax": float(THMmax),
    }

    return lam, diag


def evaluate_dose_with_J(
    dose: float,
    X_wo_dose: dict,
    Csafe: float,
    a: float,
    lambda_crit: float,
    THMmax_for_J: float
):
    r = evaluate_dose_raw(dose, X_wo_dose)

    if not r:
        return {
            "is_feasible": False,
            "J": 1e9,
            "cl_dose": float(dose),
        }

    C = r["free_cl"]
    thm = r["thm_pred"]

    denom_thm = max(float(THMmax_for_J), C_WINDOW_FOR_LAM)
    thm_term = float(thm) / denom_thm

    pen = ((C - Csafe) / max(Csafe, C_WINDOW_FOR_LAM)) ** 2

    J = float((1.0 - a) * thm_term + a * lambda_crit * pen)

    out = dict(r)
    out.update({
        "is_feasible": True,
        "J": J,
        "thm_term": thm_term,
        "penalty": float(pen),
        "lambda_crit": float(lambda_crit),
        "a": float(a),
    })

    return out


def solve_optimization(X_wo_dose: dict, Csafe: float, a: float):
    if not (HARD_L < Csafe < HARD_U):
        return None

    points = build_point_cloud(X_wo_dose, Csafe)
    if not points:
        return None

    lambda_crit, diag = compute_lambda_crit_marginal_area(
        points,
        Cmin=HARD_L,
        Csafe=Csafe
    )

    THMmax_for_J = (
        diag["THMmax"]
        if np.isfinite(diag.get("THMmax", np.nan))
        else max(p["thm_pred"] for p in points)
    )

    candidates = []
    for p in points:
        r = evaluate_dose_with_J(
            p["cl_dose"],
            X_wo_dose,
            Csafe,
            a,
            lambda_crit,
            THMmax_for_J
        )
        if r and r.get("is_feasible", False):
            candidates.append(r)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["J"])
    top = candidates[:TOPK_CAND]

    def obj(d):
        r = evaluate_dose_with_J(
            d,
            X_wo_dose,
            Csafe,
            a,
            lambda_crit,
            THMmax_for_J
        )
        return r["J"] if (r and r.get("is_feasible", False)) else 1e9

    refined = []

    for seed in top:
        seed_dose = seed["cl_dose"]
        lo = max(DOSE_MIN, seed_dose - 1.0)
        hi = min(DOSE_MAX, seed_dose + 1.0)

        res = minimize_scalar(
            obj,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": XATOL, "maxiter": MAXITER},
        )

        r = evaluate_dose_with_J(
            res.x,
            X_wo_dose,
            Csafe,
            a,
            lambda_crit,
            THMmax_for_J
        )

        if r and r.get("is_feasible", False):
            r["_lambda_diag_n"] = diag["n_used"]
            r["_A_thm"] = diag["A_thm"]
            r["_A_pen"] = diag["A_pen"]
            r["_THMmin_band"] = diag["THMmin"]
            r["_THMmax_band"] = diag["THMmax"]
            refined.append(r)

    if not refined:
        return None

    best = min(refined, key=lambda x: x["J"])
    return best


def batch_process(input_file: str, output_file: str, sheet_name: str):
    print(f"Starting batch processing: {input_file} [{sheet_name}]")

    df = pd.read_excel(input_file, sheet_name=sheet_name)

    feature_cols = ["pH", "UV254", "DOC", "TN", "Br", "EEM_I", "EEM_V", "EEM_II"]
    missing = [c for c in feature_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    results = []
    total = len(df)

    for idx, row in df.iterrows():
        print(f"Processing progress: {idx + 1}/{total}", end="\r")

        X = {c: float(row[c]) for c in feature_cols}

        sol = solve_optimization(X, CSAFE, A_WEIGHT)

        out = {
            "ID": idx + 1,
            "Csafe": CSAFE,
            "a": A_WEIGHT,
            "Cmin(HARD_L)": HARD_L,
            **X,
        }

        if sol:
            out.update({
                "Optimal_Dose": sol["cl_dose"],
                "Pred_Free_Cl": sol["free_cl"],
                "Pred_Cl_Consumed": sol.get("cl_consumed", np.nan),
                "Pred_THM": sol.get("thm_pred", np.nan),
                "Lambda_Crit": sol.get("lambda_crit", np.nan),
                "THM_term": sol.get("thm_term", np.nan),
                "Penalty": sol.get("penalty", np.nan),
                "Final_J": sol.get("J", np.nan),
                "n_points_in_[Cmin,Csafe]": sol.get("_lambda_diag_n", 0),
                "Area_THM_marginal": sol.get("_A_thm", np.nan),
                "Area_penalty": sol.get("_A_pen", np.nan),
                "THMmin_band": sol.get("_THMmin_band", np.nan),
                "THMmax_band": sol.get("_THMmax_band", np.nan),
            })

            if DIAG_PRINT and (idx < 5):
                print(
                    f"\n[Diag #{idx + 1}] lambda_crit={out['Lambda_Crit']:.4g}, "
                    f"A_thm={out['Area_THM_marginal']:.3g}, "
                    f"A_pen={out['Area_penalty']:.3g}, "
                    f"C*={out['Pred_Free_Cl']:.3f}, "
                    f"THM_term={out['THM_term']:.3f}, "
                    f"J={out['Final_J']:.3f}"
                )
        else:
            out["Optimal_Dose"] = "No Solution"

        results.append(out)

    out_df = pd.DataFrame(results)
    out_df.to_excel(output_file, index=False)

    print(f"\nProcessing completed. Results saved to: {output_file}")

    return out_df


if __name__ == "__main__":
    INPUT_EXCEL = "Dataset.xlsx"
    SHEET = "Sheet1"

    OUTPUT_EXCEL = f"Br_model_comparison_C{CSAFE}_a{A_WEIGHT}_.xlsx"
    batch_process(INPUT_EXCEL, OUTPUT_EXCEL, SHEET)