"""
predict_Cl2consumption.py
    from predict_Cl2consumption import predict_Cl2_consumpution
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from typing import Dict, Optional, List
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, root_mean_squared_error

from catboost import CatBoostRegressor
from bayes_opt import BayesianOptimization


# ======================================================
def _ensure_feature_order(X: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_names if c not in X.columns]
    extra = [c for c in X.columns if c not in feature_names]

    if missing:
        raise ValueError(f"miss: {missing}")
    if extra:
        raise ValueError(f"unnecessary: {extra}")

    return X.loc[:, feature_names].copy()


# ======================================================
# 1) train
# ======================================================
class CatBoostModelTrainer:
    def __init__(
        self,
        data_path: str,
        sheet_name: str = "chlorine",
        random_state: int = 42,
        n_folds: int = 10
    ):
        self.data_path = data_path
        self.sheet_name = sheet_name
        self.random_state = random_state
        self.n_folds = n_folds

        self.feature_cols = [
            'pH', 'UV254', 'DOC', 'TN', 'Br',
            'Chlorine dose', 'EEM_I', 'EEM_V',
        ]
        self.target_col = 'Chlorine consumption'

        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None


        self.X_train_tune: Optional[pd.DataFrame] = None
        self.X_test_tune: Optional[pd.DataFrame] = None
        self.y_train_tune: Optional[pd.Series] = None
        self.y_test_tune: Optional[pd.Series] = None

        # 最终模型训练/评估专用划分
        self.X_train_final: Optional[pd.DataFrame] = None
        self.X_test_final: Optional[pd.DataFrame] = None
        self.y_train_final: Optional[pd.Series] = None
        self.y_test_final: Optional[pd.Series] = None

        self.model: Optional[CatBoostRegressor] = None
        self.best_params: Optional[Dict] = None
        self.metrics: Optional[Dict] = None

        np.random.seed(random_state)

    # =========================

    # =========================
    def load_and_preprocess(self) -> None:
        self.df = pd.read_excel(self.data_path, sheet_name=self.sheet_name)

        X = self.df[self.feature_cols].copy()
        y = self.df[self.target_col].copy()


        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        mask = X.notna().all(axis=1) & y.notna()

        self.X = X.loc[mask].reset_index(drop=True)
        self.y = y.loc[mask].reset_index(drop=True)

        print(f" file name: {self.data_path}")
        print(f" Sheet name: {self.sheet_name}")
        print(f" data shape: {self.X.shape}")
        print(f" feature: {self.feature_cols}")
        print(f" target: {self.target_col}")


        self.X_train_tune, self.X_test_tune, self.y_train_tune, self.y_test_tune = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        print(f" train: {self.X_train_tune.shape}")
        print(f" test: {self.X_test_tune.shape}")

    # =========================
    #
    # =========================
    def _catboost_cv(
        self,
        iterations,
        depth,
        learning_rate,
        l2_leaf_reg,
        border_count,
        random_strength,
        bagging_temperature
    ):
        params = {
            'iterations': int(iterations),
            'depth': int(depth),
            'learning_rate': float(learning_rate),
            'l2_leaf_reg': float(l2_leaf_reg),
            'border_count': int(border_count),
            'random_strength': float(random_strength),
            'bagging_temperature': float(bagging_temperature),
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': False
        }

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in kf.split(self.X_train_tune):
            X_tr = self.X_train_tune.iloc[train_idx]
            X_val = self.X_train_tune.iloc[val_idx]
            y_tr = self.y_train_tune.iloc[train_idx]
            y_val = self.y_train_tune.iloc[val_idx]

            model = CatBoostRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )

            pred = model.predict(X_val)
            scores.append(r2_score(y_val, pred))

        return float(np.mean(scores))


    # =========================
    def optimize_hyperparameters(self) -> Dict:
        if self.X_train_tune is None or self.y_train_tune is None:
            raise ValueError(" load_and_preprocess() ")

        pbounds = {
            'iterations': (100, 500),
            'depth': (3, 6),
            'learning_rate': (0.01, 0.08),
            'l2_leaf_reg': (3, 15),
            'border_count': (64, 255),
            'random_strength': (1, 10),
            'bagging_temperature': (0, 5)
        }

        optimizer = BayesianOptimization(
            f=self._catboost_cv,
            pbounds=pbounds,
            random_state=42
        )

        optimizer.maximize(init_points=5, n_iter=20)

        self.best_params = optimizer.max['params']
        self.best_params['iterations'] = int(self.best_params['iterations'])
        self.best_params['depth'] = int(self.best_params['depth'])
        self.best_params['border_count'] = int(self.best_params['border_count'])

        print("\n best_params:")
        for k, v in self.best_params.items():
            print(f"{k}: {v}")

        print(f"\n 10CV R²: {optimizer.max['target']:.4f}")

        return self.best_params


    # =========================
    def train_final_model(self, params: Optional[Dict] = None) -> CatBoostRegressor:
        if self.X is None or self.y is None:
            raise ValueError(" load_and_preprocess() ")

        if params is None:
            if self.best_params is None:
                self.optimize_hyperparameters()
            params = self.best_params


        self.X_train_final, self.X_test_final, self.y_train_final, self.y_test_final = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )

        self.model = CatBoostRegressor(
            **params,
            random_seed=42,
            early_stopping_rounds=50,
            verbose=0
        )

        self.model.fit(
            self.X_train_final, self.y_train_final,
            eval_set=(self.X_test_final, self.y_test_final),
            use_best_model=True,
            verbose=0
        )

        print("\n final model finished")
        return self.model

    # =========================

    # =========================
    def evaluate(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError(" train_final_model() ")

        y_train_pred = self.model.predict(self.X_train_final)
        y_test_pred = self.model.predict(self.X_test_final)

        self.metrics = {
            'train_R2': float(r2_score(self.y_train_final, y_train_pred)),
            'train_RMSE': float(root_mean_squared_error(self.y_train_final, y_train_pred)),
            'test_R2': float(r2_score(self.y_test_final, y_test_pred)),
            'test_RMSE': float(root_mean_squared_error(self.y_test_final, y_test_pred))
        }

        print("\n result:")
        print(f"Train R²: {self.metrics['train_R2']:.4f}")
        print(f"Train RMSE: {self.metrics['train_RMSE']:.4f}")
        print(f"Test R²: {self.metrics['test_R2']:.4f}")
        print(f"Test RMSE: {self.metrics['test_RMSE']:.4f}")

        return self.metrics


    # =========================
    def save_results_excel(self, output_path: str = 'catboost_results.xlsx') -> str:
        if self.metrics is None:
            raise ValueError(" evaluate() ")

        results_df = pd.DataFrame([self.metrics])

        with pd.ExcelWriter(output_path) as writer:
            results_df.to_excel(writer, sheet_name='evaluation', index=False)

        print(f"\n {output_path}")
        return output_path


    # =========================
    def save_model(self, directory: str = 'saved_models') -> Dict[str, str]:
        if self.model is None:
            raise ValueError(" train_final_model() ")

        os.makedirs(directory, exist_ok=True)

        model_paths = {
            'catboost': os.path.join(directory, 'catboost_model.cbm'),
            'joblib': os.path.join(directory, 'catboost_model.pkl'),
            'metadata': os.path.join(directory, 'model_metadata.json')
        }

        self.model.save_model(model_paths['catboost'])
        joblib.dump(self.model, model_paths['joblib'])

        metadata = {
            'best_params': self.best_params,
            'feature_names': self.feature_cols,
            'target_name': self.target_col,
            'sheet_name': self.sheet_name,
            'data_path': self.data_path,
            'random_state': self.random_state,
            'n_folds': self.n_folds
        }

        with open(model_paths['metadata'], 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        print(f"\n save to: {directory}")
        print(f"  cbm: {model_paths['catboost']}")
        print(f"  pkl: {model_paths['joblib']}")
        print(f"  meta: {model_paths['metadata']}")

        return model_paths


    # =========================
    def full_pipeline(self) -> Dict:
        print("...")
        print("=" * 60)

        self.load_and_preprocess()
        self.optimize_hyperparameters()
        self.train_final_model()
        self.evaluate()
        self.save_results_excel()
        model_paths = self.save_model()

        print("=" * 60)
        print("")

        return {
            'model': self.model,
            'best_params': self.best_params,
            'metrics': self.metrics,
            'paths': model_paths
        }


# ======================================================

def predict_Cl2_consumpution(
    X: pd.DataFrame,
    model_path: str = "saved_models/catboost_model.pkl",
    metadata_path: str = "saved_models/model_metadata.json"
):
    """

        from predict_Cl2consumption import predict_Cl2_consumpution
        y_pred = predict_Cl2_consumpution(X)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"miss: {model_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"miss: {metadata_path}")

    model = joblib.load(model_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = meta["feature_names"]
    X_use = _ensure_feature_order(X, feature_names)

    return model.predict(X_use)


if __name__ == "__main__":
    trainer = CatBoostModelTrainer(
        data_path='Dataset.xlsx',
        sheet_name='chlorine',
        n_folds=10,
        random_state=42
    )
    _ = trainer.full_pipeline()