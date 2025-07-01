import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # Added r2_score
import xgboost as xgb
import joblib
import os

# Load your preprocessed features
FEATURES_PATH = '../dataset-primitively-engineered/score_prediction_features.csv'
TH_PATH = '../dataset-primitively-engineered/th-public.csv'

features = pd.read_csv(FEATURES_PATH)
loaded_df = pd.read_csv(TH_PATH)
loaded_df = loaded_df.rename(columns={'hash': 'student_id', 'TH': 'exam_score'})
loaded_df['exam_score'] = pd.to_numeric(loaded_df['exam_score'], errors='coerce')
merged = pd.merge(features, loaded_df, on='student_id', how='inner')
merged = merged.dropna(subset=['exam_score'])

X = merged.drop(columns=['student_id', 'class_id', 'exam_score'])
y = merged['exam_score']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def objective(trial):
    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "tree_method": "gpu_hist",
        "device": "cuda",
        "predictor": "gpu_predictor",
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),
        "gamma": trial.suggest_float("gamma", 0, 5),
    }
    if param["booster"] == "dart":
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    evals = [(dvalid, "eval")]

    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=False
    )
    preds = bst.predict(dvalid)
    
    # Calculate multiple metrics
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    mse = mean_squared_error(y_valid, preds)
    r2 = r2_score(y_valid, preds)
    
    # Store additional metrics in trial
    trial.set_user_attr("mse", mse)
    trial.set_user_attr("r2", r2)
    
    return -rmse  # Optuna maximizes, so we minimize RMSE by maximizing negative RMSE

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500, timeout=1800)  # 50 trials or 30 minutes

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    # Retrieve all stored metrics
    print("  RMSE: {:.4f}".format(-trial.value))
    print("  MSE: {:.4f}".format(trial.user_attrs["mse"]))
    print("  R²: {:.4f}".format(trial.user_attrs["r2"]))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Train final model with best params
    best_params = trial.params
    best_params.update({
        "verbosity": 0,
        "objective": "reg:squarederror",
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor"
    })
    dtrain = xgb.DMatrix(X, label=y)
    final_model = xgb.train(best_params, dtrain, num_boost_round=200)
    
    # Evaluate final model on entire dataset
    final_preds = final_model.predict(dtrain)
    final_rmse = np.sqrt(mean_squared_error(y, final_preds))
    final_mse = mean_squared_error(y, final_preds)
    final_r2 = r2_score(y, final_preds)
    
    print("\nFinal Model Performance:")
    print(f"  RMSE: {final_rmse:.4f}")
    print(f"  MSE: {final_mse:.4f}")
    print(f"  R²: {final_r2:.4f}")
    
    joblib.dump(final_model, "student_score_predictor_optuna_xgb.pkl")
    print("Best model saved to student_score_predictor_optuna_xgb.pkl")
