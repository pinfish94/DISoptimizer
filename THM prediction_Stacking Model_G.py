import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, root_mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import shap


df = pd.read_excel('Dataset.xlsx', sheet_name='THM')
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



def get_best_models():

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



best_xgb_params, best_lgb_params = get_best_models()



print("XGBoost best parameter:")
for param, value in best_xgb_params.items():
    print(f"  {param}: {value}")

print("\nLightGBM best parameter:")
for param, value in best_lgb_params.items():
    print(f"  {param}: {value}")


class StackingModel:
    def __init__(self, base_models, kfold=5):
        self.base_models = base_models
        self.kfold = kfold
        self.meta_model = None
        self.model_weights = None

    def fit(self, X, y):

        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        kf = KFold(n_splits=self.kfold, shuffle=True, random_state=42)


        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]

                model.fit(X_train, y_train)
                meta_features[val_idx, i] = model.predict(X_val)


            model.fit(X, y)


        r2_scores = []
        for i in range(len(self.base_models)):
            r2 = r2_score(y, meta_features[:, i])
            r2_scores.append(r2)


        total = sum(r2_scores)
        self.model_weights = [score / total for score in r2_scores]
        print(f"Model Weighting: {self.model_weights}")

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            predictions[:, i] = model.predict(X)


        weighted_pred = np.zeros(X.shape[0])
        for i in range(len(self.base_models)):
            weighted_pred += predictions[:, i] * self.model_weights[i]

        return weighted_pred



xgb_model = xgb.XGBRegressor(**best_xgb_params)
lgb_model = lgb.LGBMRegressor(**best_lgb_params)

stacking_model = StackingModel(base_models=[xgb_model, lgb_model])
stacking_model.fit(X_train, y_train)


print("\n=== model configuration ===")
print("XGBoost  configuration:")
print(xgb_model)

print("\nLightGBM configuration:")
print(lgb_model)



def detailed_evaluation(model, X_train, y_train, X_test, y_test, model_name=""):

    train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, train_pred)
    train_rmse = root_mean_squared_error(y_train, train_pred)


    test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = root_mean_squared_error(y_test, test_pred)

    print(f"\n=== {model_name} performance ===")
    print(f"Train R²: {train_r2:.4f} | RMSE: {train_rmse:.4f}")
    print(f"Test R²: {test_r2:.4f} | RMSE: {test_rmse:.4f}")


    return {
        'train': {'true': y_train, 'pred': train_pred},
        'test': {'true': y_test, 'pred': test_pred}
    }



stacking_results = detailed_evaluation(
    stacking_model, X_train, y_train, X_test, y_test, "stacking model"
)


xgb_results = detailed_evaluation(
    xgb_model, X_train, y_train, X_test, y_test, "XGBoost"
)
lgb_results = detailed_evaluation(
    lgb_model, X_train, y_train, X_test, y_test, "LightGBM"
)



results_df = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'Stacking'],
    'Train R²': [
        r2_score(y_train, xgb_results["train"]["pred"]),
        r2_score(y_train, lgb_results["train"]["pred"]),
        r2_score(y_train, stacking_results["train"]["pred"])
    ],
    'Train RMSE': [
        root_mean_squared_error(y_train, xgb_results["train"]["pred"]),
        root_mean_squared_error(y_train, lgb_results["train"]["pred"]),
        root_mean_squared_error(y_train, stacking_results["train"]["pred"])
    ],
    'Test R²': [
        r2_score(y_test, xgb_results["test"]["pred"]),
        r2_score(y_test, lgb_results["test"]["pred"]),
        r2_score(y_test, stacking_results["test"]["pred"])
    ],
    'Test RMSE': [
        root_mean_squared_error(y_test, xgb_results["test"]["pred"]),
        root_mean_squared_error(y_test, lgb_results["test"]["pred"]),
        root_mean_squared_error(y_test, stacking_results["test"]["pred"])
    ]
})

print("\nm model performance table:")
print(results_df.to_markdown(index=False))



