import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, root_mean_squared_error

from catboost import CatBoostRegressor
from bayes_opt import BayesianOptimization

df = pd.read_excel('Dataset.xlsx', sheet_name="chlorine")

feature_cols = [
    'pH', 'UV254', 'DOC', 'TN', 'Br',
    'Chlorine dose', 'EEM_I', 'EEM_V'
]

target_col = 'Chlorine consumption'

X = df[feature_cols].copy()
y = df[target_col].copy()

X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)

X_train_full, X_test_holdout, y_train_full, y_test_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def catboost_cv(iterations, depth, learning_rate, l2_leaf_reg,
                border_count, random_strength, bagging_temperature):

    params = {
        'iterations': int(iterations),
        'depth': int(depth),
        'learning_rate': learning_rate,
        'l2_leaf_reg': l2_leaf_reg,
        'border_count': int(border_count),
        'random_strength': random_strength,
        'bagging_temperature': bagging_temperature,
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': False,
        'allow_writing_files': False
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_train_full):
        X_tr, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        model = CatBoostRegressor(**params)

        model.fit(
            X_tr,
            y_tr,
            verbose=False
        )

        pred = model.predict(X_val)
        scores.append(r2_score(y_val, pred))

    return np.mean(scores)

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
    f=catboost_cv,
    pbounds=pbounds,
    random_state=42
)

optimizer.maximize(init_points=5, n_iter=20)

best_params = optimizer.max['params']

best_params['iterations'] = int(best_params['iterations'])
best_params['depth'] = int(best_params['depth'])
best_params['border_count'] = int(best_params['border_count'])

print("\nBest parameters:")
for k, v in best_params.items():
    print(f"{k}: {v}")

print(f"\nBest CV R2: {optimizer.max['target']:.4f}")

seeds = [2, 22, 42, 62, 82]
results = []

for seed in seeds:
    print(f"\nCurrent random seed: {seed}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = CatBoostRegressor(
        **best_params,
        random_seed=42,
        verbose=0,
        loss_function='RMSE',
        allow_writing_files=False
    )

    model.fit(
        X_train,
        y_train,
        verbose=0
    )

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)

    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)

    print(f"Train R2: {r2_train:.4f} | Test R2: {r2_test:.4f}")

    results.append({
        'seed': seed,
        'train_R2': r2_train,
        'train_RMSE': rmse_train,
        'test_R2': r2_test,
        'test_RMSE': rmse_test
    })

results_df = pd.DataFrame(results)
summary = results_df.describe().loc[['mean', 'std']]

print("\nResults summary:")
print(summary)

with pd.ExcelWriter('catboost_results.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='raw_results', index=False)
    summary.to_excel(writer, sheet_name='summary')

print("\nResults saved to catboost_results_no_early_stopping.xlsx")