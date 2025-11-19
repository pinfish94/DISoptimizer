import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, root_mean_squared_error
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score


df = pd.read_excel('Dataset.xlsx', sheet_name="THM")


X = df.iloc[:, 1:11]
y = df.iloc[:, 11]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=62
)


def lgb_cv(num_leaves, max_depth, learning_rate, n_estimators, min_child_samples):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': int(num_leaves),
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'min_child_samples': int(min_child_samples),
        'random_state': 42,
        'verbosity': -1
    }

    model = lgb.LGBMRegressor(**params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    return np.mean(cv_scores)


pbounds = {
    'num_leaves': (20, 100),
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.1),
    'n_estimators': (50, 200),
    'min_child_samples': (5, 50)
}


optimizer = BayesianOptimization(
    f=lgb_cv,
    pbounds=pbounds,
    random_state=42,
)

optimizer.maximize(
    init_points=5,
    n_iter=20,
)


best_params = optimizer.max['params']
best_params['num_leaves'] = int(best_params['num_leaves'])
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['min_child_samples'] = int(best_params['min_child_samples'])
print("Best parameter:", best_params)




final_model = lgb.LGBMRegressor(
    **best_params,
    random_state=42,
    verbosity=-1
)


kf = KFold(n_splits=10, shuffle=True, random_state=42)
r2_scores = []
rmse_scores = []

for train_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    final_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    y_val_pred = final_model.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    rmse = root_mean_squared_error(y_val, y_val_pred)

    r2_scores.append(r2)
    rmse_scores.append(rmse)

print(f"cross validation R² average: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"cross validation RMSE average: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")


final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(50, verbose=False)]
)


y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)


r2_train = r2_score(y_train, y_train_pred)
rmse_train = root_mean_squared_error(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_test = root_mean_squared_error(y_test, y_test_pred)

print(f"Train R²: {r2_train:.4f} | RMSE: {rmse_train:.4f}")
print(f"Test R²: {r2_test:.4f} | RMSE: {rmse_test:.4f}")
