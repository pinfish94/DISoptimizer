import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


df = pd.read_excel('Dataset.xlsx', sheet_name="THM")


X = df.iloc[:, 1:10]
y = df.iloc[:, 10]


X = X.apply(pd.to_numeric, errors='coerce')

X = X.dropna()
y = y[X.index]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=52
)


def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'min_samples_split': int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf),
        'max_features': min(max_features, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }

    model = RandomForestRegressor(**params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    return np.mean(cv_scores)



pbounds = {
    'n_estimators': (50, 200),
    'max_depth': (2, 20),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 5),
    'max_features': (0.1, 0.999)  # 特征选择比例
}


optimizer = BayesianOptimization(
    f=rf_cv,
    pbounds=pbounds,
    random_state=42,
)

optimizer.maximize(
    init_points=5,
    n_iter=20,
)


best_params = optimizer.max['params']

best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_samples_split'] = int(best_params['min_samples_split'])
best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
print("Best parameter:", best_params)


final_model = RandomForestRegressor(
    **best_params,
    random_state=42,
    n_jobs=-1
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
