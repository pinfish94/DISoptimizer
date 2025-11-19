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


df = pd.read_excel('Dataset.xlsx', sheet_name="chlorine")


X = df.iloc[:, 1:10]
y = df.iloc[:, 10]


X = X.apply(pd.to_numeric, errors='coerce')

X = X.dropna()
y = y[X.index]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


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
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    return np.mean(cv_scores)


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

optimizer.maximize(
    init_points=5,
    n_iter=20,
)

best_params = optimizer.max['params']

best_params['iterations'] = int(best_params['iterations'])
best_params['depth'] = int(best_params['depth'])
best_params['border_count'] = int(best_params['border_count'])
print("Best parameter:", best_params)


final_model = CatBoostRegressor(
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

    final_model.fit(X_tr, y_tr)
    y_val_pred = final_model.predict(X_val)

    r2 = r2_score(y_val, y_val_pred)
    rmse = root_mean_squared_error(y_val, y_val_pred)

    r2_scores.append(r2)
    rmse_scores.append(rmse)

print(f"cross validation R² average: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"cross validation RMSE average: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

final_model.fit(X_train, y_train)

y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
rmse_train = root_mean_squared_error(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_test = root_mean_squared_error(y_test, y_test_pred)

print(f"Train R²: {r2_train:.4f} | RMSE: {rmse_train:.4f}")
print(f"Test R²: {r2_test:.4f} | RMSE: {rmse_test:.4f}")


