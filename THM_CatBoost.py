import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from catboost import CatBoostRegressor
from bayes_opt import BayesianOptimization


# =========================
df = pd.read_excel('Dataset.xlsx', sheet_name='THM')

# =========================

feature_cols = [
    'pH',
    'UV254',
    'DOC',
    'TN',
    'Br',
    'Chlorine dose',
    'Chlorine consumption',
    'EEM_II',

]
# 'chlorine consumption',

target_col = 'THM4'


missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
if missing_cols:
    raise ValueError(
        f'miss: {missing_cols}\n'
        f'columns_name: {list(df.columns)}'
    )

X = df[feature_cols].copy()
y = df[target_col].copy()


# =========================
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')

data = pd.concat([X, y], axis=1).dropna()
X = data[feature_cols]
y = data[target_col]

print(f'n: {len(X)}')
print(f'features: {X.shape[1]}')

# =========================

X_train_42, X_test_42, y_train_42, y_test_42 = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
def cb_cv(iterations, depth, learning_rate, l2_leaf_reg,
          border_count, random_strength, bagging_temperature):
    params = {
        'iterations': int(round(iterations)),
        'depth': int(round(depth)),
        'learning_rate': learning_rate,
        'l2_leaf_reg': l2_leaf_reg,
        'border_count': int(round(border_count)),
        'random_strength': random_strength,
        'bagging_temperature': bagging_temperature,
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': False
    }

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_train_42):
        X_tr, X_val = X_train_42.iloc[train_idx], X_train_42.iloc[val_idx]
        y_tr, y_val = y_train_42.iloc[train_idx], y_train_42.iloc[val_idx]

        model = CatBoostRegressor(**params)

        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            use_best_model=True,
            verbose=False
        )

        y_val_pred = model.predict(X_val)
        scores.append(r2_score(y_val, y_val_pred))

    return np.mean(scores)

# =========================

pbounds = {
    'iterations': (100, 500),
    'depth': (3, 6),
    'learning_rate': (0.01, 0.08),
    'l2_leaf_reg': (3, 15),
    'border_count': (64, 255),
    'random_strength': (1, 10),
    'bagging_temperature': (0, 5)
}

# =========================

optimizer = BayesianOptimization(
    f=cb_cv,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(
    init_points=5,
    n_iter=20
)

# =========================

best_params = optimizer.max['params']
best_params['iterations'] = int(round(best_params['iterations']))
best_params['depth'] = int(round(best_params['depth']))
best_params['border_count'] = int(round(best_params['border_count']))

print('\n' + '=' * 60)
print('CatBoost Bayesian')
print('=' * 60)
print(f"10 CV R²: {optimizer.max['target']:.4f}")
print('best_params:')
for k, v in best_params.items():
    print(f'{k}: {v}')
print('=' * 60)

# =========================

seeds = [2, 22, 42, 62, 82]
results = []

seed42_train_pred_df = None
seed42_test_pred_df = None

for seed in seeds:
    print(f'\nevaluation random_state = {seed}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = CatBoostRegressor(
        **best_params,
        loss_function='RMSE',
        random_seed=42,
        early_stopping_rounds=50,
        verbose=False
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        use_best_model=True,
        verbose=False
    )

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)

    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

    best_iter = model.get_best_iteration()

    print(
        f'seed={seed} | '
        f'train_R2={train_r2:.4f}, train_RMSE={train_rmse:.4f}, '
        f'test_R2={test_r2:.4f}, test_RMSE={test_rmse:.4f}, '
        f'best_iteration={best_iter}'
    )

    results.append({
        'seed': seed,
        'best_iteration': best_iter,
        'train_R2': train_r2,
        'train_RMSE': train_rmse,
        'test_R2': test_r2,
        'test_RMSE': test_rmse
    })


    if seed == 42:
        seed42_train_pred_df = pd.DataFrame({
            'Actual_THM4': y_train.values,
            'Predicted_THM4': y_train_pred
        })

        seed42_test_pred_df = pd.DataFrame({
            'Actual_THM4': y_test.values,
            'Predicted_THM4': y_test_pred
        })

# =========================

results_df = pd.DataFrame(results)

summary_df = pd.DataFrame({
    'metric': ['train_R2', 'train_RMSE', 'test_R2', 'test_RMSE'],
    'mean': [
        results_df['train_R2'].mean(),
        results_df['train_RMSE'].mean(),
        results_df['test_R2'].mean(),
        results_df['test_RMSE'].mean()
    ],
    'std': [
        results_df['train_R2'].std(ddof=1),
        results_df['train_RMSE'].std(ddof=1),
        results_df['test_R2'].std(ddof=1),
        results_df['test_RMSE'].std(ddof=1)
    ]
})

best_params_df = pd.DataFrame(
    [{'parameter': k, 'value': v} for k, v in best_params.items()]
)

if seed42_train_pred_df is None or seed42_test_pred_df is None:
    raise ValueError('fail seed=42 。')

# =========================

output_file = 'catboost_THM4_results.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='seed_results', index=False)
    summary_df.to_excel(writer, sheet_name='summary', index=False)
    best_params_df.to_excel(writer, sheet_name='best_params', index=False)
    seed42_train_pred_df.to_excel(writer, sheet_name='seed42_train_pred', index=False)
    seed42_test_pred_df.to_excel(writer, sheet_name='seed42_test_pred', index=False)

print(f'\nsave to: {output_file}')
print(':')
print('- seed42_train_pred: seed 42')
print('- seed42_test_pred: seed 42')