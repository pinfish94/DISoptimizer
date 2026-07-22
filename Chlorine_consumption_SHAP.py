import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from catboost import CatBoostRegressor
from bayes_opt import BayesianOptimization

# =========================

def save_shap_beeswarm_and_importance(
    model,
    X_train,
    X_sample,
    save_path,
    random_state=42
):
    REQUIRED_COLUMNS = list(X_train.columns)

    X_train = X_train[REQUIRED_COLUMNS].copy()
    X_sample = X_sample[REQUIRED_COLUMNS].copy()


    n_background = min(200, len(X_train))
    X_background = X_train.sample(n=n_background, random_state=random_state).copy()

    def _predict_fn(data):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=REQUIRED_COLUMNS)
        elif not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(np.asarray(data), columns=REQUIRED_COLUMNS)
        else:
            data = data[REQUIRED_COLUMNS].copy()

        return np.asarray(model.predict(data))

    explainer = shap.Explainer(_predict_fn, X_background, algorithm="permutation")
    shap_values = explainer(X_sample, max_evals=2 * X_sample.shape[1] + 1)

    # ==============================

    shap_array = shap_values.values
    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    contribution_percent = mean_abs_shap / mean_abs_shap.sum() * 100

    shap_importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap,
        "contribution_percent": contribution_percent
    }).sort_values("mean_abs_shap", ascending=False)

    output_dir = Path(save_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    excel_path = output_dir / "shap_global_contribution_percent.xlsx"
    csv_path = output_dir / "shap_global_contribution_percent.csv"

    shap_importance_df.to_excel(excel_path, index=False)
    shap_importance_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n contribution：")
    print(shap_importance_df)



    # ==============================
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_array,
        X_sample,
        plot_type="dot",
        cmap="viridis_r",  # 黄-绿-蓝-紫色系
        show=False
    )

    fig = plt.gcf()
    axes = fig.axes
    ax = axes[0]


    axis_label_size = 25
    tick_label_size = 22
    cbar_label_size = 30
    cbar_tick_size = 30



    ax.xaxis.label.set_fontname("Arial")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontname("Arial")
    ax.yaxis.label.set_fontweight("bold")


    for label in ax.get_xticklabels():
        label.set_fontname("Arial Black")  #

    for label in ax.get_yticklabels():
        label.set_fontname("Arial Black")  #


    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(axis="x", width=1.5)


    if "left" in ax.spines:
        ax.spines["left"].set_linewidth(1.5)
    if "top" in ax.spines:
        ax.spines["top"].set_linewidth(1.5)
    if "right" in ax.spines:
        ax.spines["right"].set_linewidth(1.5)


    if len(axes) > 1:
        cbar_ax = axes[1]
        cbar_ax.yaxis.label.set_fontname("Arial")
        cbar_ax.yaxis.label.set_fontweight("bold")

        for label in cbar_ax.get_yticklabels():
            label.set_fontname("Arial")
            label.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()

    print(f"\n🧠 figure save to : {save_path}（sample number {len(X_sample)} ）")

    return save_path


# =========================

# =========================
df = pd.read_excel('Dataset.xlsx', sheet_name="chlorine")

# =========================

feature_cols = [
    'pH', 'UV254', 'DOC', 'TN', 'Br',
    'Chlorine dose', 'EEM_I', 'EEM_V',
]


target_col = 'Chlorine consumption'


missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
if missing_cols:
    raise ValueError(
        f'miss: {missing_cols}\n'
        f'column name: {list(df.columns)}'
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
print(f'feature: {X.shape[1]}')



# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
def catboost_cv(iterations, depth, learning_rate, l2_leaf_reg,
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

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = CatBoostRegressor(**params)

        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            use_best_model=True,
            verbose=False
        )

        pred = model.predict(X_val)
        scores.append(r2_score(y_val, pred))

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

optimizer = BayesianOptimization(
    f=catboost_cv,
    pbounds=pbounds,
    random_state=42
)

optimizer.maximize(init_points=5, n_iter=20)

# =========================

best_params = optimizer.max['params']

best_params['iterations'] = int(round(best_params['iterations']))
best_params['depth'] = int(round(best_params['depth']))
best_params['border_count'] = int(round(best_params['border_count']))

print("\n best params:")
for k, v in best_params.items():
    print(f"{k}: {v}")

print(f"\n 10 CV R²: {optimizer.max['target']:.4f}")

# =========================

seeds = [2, 22, 42, 62, 82]
results = []

seed42_train_pred_df = None
seed42_test_pred_df = None

for seed in seeds:
    print(f"\n : {seed}")


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = CatBoostRegressor(
        **best_params,
        loss_function='RMSE',
        random_seed=42,
        early_stopping_rounds=50,
        verbose=0
    )


    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True,
        verbose=0
    )


    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)


    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)

    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)

    print(f"Train R²: {r2_train:.4f} | Test R²: {r2_test:.4f}")

    results.append({
        'seed': seed,
        'train_R2': r2_train,
        'train_RMSE': rmse_train,
        'test_R2': r2_test,
        'test_RMSE': rmse_test
    })


    if seed == 42:
        seed42_train_pred_df = pd.DataFrame({
            'Actual_chlorine_consumption': y_train.values,
            'Predicted_chlorine_consumption': y_train_pred
        })

        seed42_test_pred_df = pd.DataFrame({
            'Actual_chlorine_consumption': y_test.values,
            'Predicted_chlorine_consumption': y_test_pred
        })

        # =========================

        # =========================
        n_sample = min(100, len(X_test))
        X_sample = X_test.sample(n=n_sample, random_state=42).copy()

        shap_plot_path = 'shap_summary_plot_chlorine_consumption_seed42.svg'
        save_shap_beeswarm_and_importance(
            model=model,
            X_train=X_train,
            X_sample=X_sample,
            save_path=shap_plot_path,
            random_state=42
        )

# =========================

results_df = pd.DataFrame(results)
summary = results_df.describe().loc[['mean', 'std']]

print("\n result:")
print(summary)

# =========================


with pd.ExcelWriter('catboost_results.xlsx', engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='raw_results', index=False)
    summary.to_excel(writer, sheet_name='summary')

    if seed42_train_pred_df is not None:
        seed42_train_pred_df.to_excel(writer, sheet_name='seed42_train_pred', index=False)

    if seed42_test_pred_df is not None:
        seed42_test_pred_df.to_excel(writer, sheet_name='seed42_test_pred', index=False)

print("\n results save to catboost_results.xlsx")
