import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from catboost import CatBoostRegressor, Pool
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
import seaborn as sns
import shap
import joblib
import os



def main():
    df = pd.read_excel('Dataset.xlsx', sheet_name="chlorine")

    X = df.iloc[:, 1:10]
    y = df.iloc[:, 10]

    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna()
    y = y[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    best_params = optimize_hyperparameters(X_train, y_train)
    print("best parameters:", best_params)


    final_model = train_final_model(X_train, y_train, best_params)


    evaluate_model(final_model, X_train, y_train, X_test, y_test)


    save_model(final_model)


def optimize_hyperparameters(X_train, y_train):


    def cb_cv(iterations, depth, learning_rate, l2_leaf_reg, border_count):
        params = {
            'iterations': int(iterations),
            'depth': int(depth),
            'learning_rate': learning_rate,
            'l2_leaf_reg': l2_leaf_reg,
            'border_count': int(border_count),
            'random_seed': 42,
            'verbose': False,
            'loss_function': 'RMSE'
        }
        model = CatBoostRegressor(**params)
        return cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()

    pbounds = {
        'iterations': (50, 500),
        'depth': (4, 8),
        'learning_rate': (0.01, 0.1),
        'l2_leaf_reg': (3, 15),
        'border_count': (32, 128)
    }

    optimizer = BayesianOptimization(
        f=cb_cv,
        pbounds=pbounds,
        random_state=42,
    )

    optimizer.maximize(init_points=5, n_iter=20)

    best_params = optimizer.max['params']
    best_params['iterations'] = int(best_params['iterations'])
    best_params['depth'] = int(best_params['depth'])
    best_params['border_count'] = int(best_params['border_count'])

    return best_params


def train_final_model(X_train, y_train, best_params):

    model = CatBoostRegressor(
        **best_params,
        early_stopping_rounds=20,
        random_seed=42,
        verbose=False
    )


    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scores = []
    rmse_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_val_pred = model.predict(X_val)

        r2_scores.append(r2_score(y_val, y_val_pred))
        rmse_scores.append(root_mean_squared_error(y_val, y_val_pred))

    print(f" kf  R² average: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f" kf average: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")


    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)

    print(f"train R²: {r2_train:.4f} | RMSE: {rmse_train:.4f}")
    print(f"test R²: {r2_test:.4f} | RMSE: {rmse_test:.4f}")


def save_model(model):

    os.makedirs('saved_models', exist_ok=True)
    model_path = 'saved_models/catboost_model.cbm'


    model.save_model(model_path)


    joblib.dump(model, 'saved_models/catboost_model.pkl')

    print(f"model save  {model_path}")


if __name__ == "__main__":
    main()