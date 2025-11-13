import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, root_mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import joblib
import os
from stacking_model import StackingModel


plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False


def main():

    df = pd.read_excel('Dataset.xlsx', sheet_name="THM")
    X = df.iloc[:, 1:11]
    y = df.iloc[:, 11]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    def optimize_xgb(max_depth, learning_rate, n_estimators, subsample, colsample_bytree):
        params = {
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'n_estimators': int(n_estimators),
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': 42
        }
        model = xgb.XGBRegressor(**params)
        return cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()

    def optimize_lgb(num_leaves, max_depth, learning_rate, n_estimators):
        params = {
            'num_leaves': int(num_leaves),
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'n_estimators': int(n_estimators),
            'random_state': 42
        }
        model = lgb.LGBMRegressor(**params)
        return cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()



    best_xgb_params, best_lgb_params = get_best_models(optimize_xgb, optimize_lgb)


    xgb_model = xgb.XGBRegressor(**best_xgb_params)
    lgb_model = lgb.LGBMRegressor(**best_lgb_params)


    stacking_model = StackingModel(base_models=[xgb_model, lgb_model])
    stacking_model.fit(X_train, y_train)



    stacking_results = detailed_evaluation(
        stacking_model, X_train, y_train, X_test, y_test, "stacking_model"
    )
    xgb_results = detailed_evaluation(
        xgb_model, X_train, y_train, X_test, y_test, "XGBoost"
    )
    lgb_results = detailed_evaluation(
        lgb_model, X_train, y_train, X_test, y_test, "LightGBM"
    )


    print("\nsave model...")
    save_models(stacking_model, xgb_model, lgb_model)



def get_best_models(optimize_xgb, optimize_lgb):

    xgb_optimizer = BayesianOptimization(
        f=optimize_xgb,
        pbounds={
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (50, 200),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0)
        },
        random_state=42
    )
    xgb_optimizer.maximize(init_points=5, n_iter=15)


    lgb_optimizer = BayesianOptimization(
        f=optimize_lgb,
        pbounds={
            'num_leaves': (20, 100),
            'max_depth': (3, 12),
            'learning_rate': (0.01, 0.3),
            'n_estimators': (50, 200)
        },
        random_state=42
    )
    lgb_optimizer.maximize(init_points=5, n_iter=15)


    best_xgb_params = xgb_optimizer.max['params']
    best_xgb_params['max_depth'] = int(best_xgb_params['max_depth'])
    best_xgb_params['n_estimators'] = int(best_xgb_params['n_estimators'])

    best_lgb_params = lgb_optimizer.max['params']
    best_lgb_params['num_leaves'] = int(best_lgb_params['num_leaves'])
    best_lgb_params['max_depth'] = int(best_lgb_params['max_depth'])
    best_lgb_params['n_estimators'] = int(best_lgb_params['n_estimators'])

    return best_xgb_params, best_lgb_params


def detailed_evaluation(model, X_train, y_train, X_test, y_test, model_name=""):

    train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, train_pred)
    train_rmse = root_mean_squared_error(y_train, train_pred)


    test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = root_mean_squared_error(y_test, test_pred)


    print(f"\n=== {model_name} performance ===")
    print(f"train R²: {train_r2:.4f} | RMSE: {train_rmse:.4f}")
    print(f"test R²: {test_r2:.4f} | RMSE: {test_rmse:.4f}")

    return {
        'train': {'true': y_train, 'pred': train_pred},
        'test': {'true': y_test, 'pred': test_pred}
    }


def save_models(stacking_model, xgb_model, lgb_model):

    os.makedirs('saved_models', exist_ok=True)


    joblib.dump(stacking_model, 'saved_models/stacking_model.pkl')


    joblib.dump(xgb_model, 'saved_models/xgb_model.pkl')
    joblib.dump(lgb_model, 'saved_models/lgb_model.pkl')


    with open('saved_models/model_info.txt', 'w') as f:
        f.write("model weight:\n")
        f.write(f"XGBoost: {stacking_model.model_weights[0]:.4f}\n")
        f.write(f"LightGBM: {stacking_model.model_weights[1]:.4f}\n")


if __name__ == "__main__":
    main()
