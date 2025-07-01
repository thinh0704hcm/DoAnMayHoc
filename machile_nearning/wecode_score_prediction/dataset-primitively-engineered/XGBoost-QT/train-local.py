import os
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

# Configuration
DATA_PATH = "../dataset-primitively-engineered/"
FEATURES_FILE = "score_prediction_features.csv"
TARGET_FILE = "qt-public.csv"
MODEL_PATH = "qt_student_score_predictor_optuna_xgb_local.pkl"

def train_xgboost():
    # Load data
    features = pd.read_csv(os.path.join(DATA_PATH, FEATURES_FILE))
    loaded_df = pd.read_csv(os.path.join(DATA_PATH, TARGET_FILE))
    loaded_df = loaded_df.rename(columns={'hash': 'student_id', 'diemqt': 'exam_score'})
    loaded_df['exam_score'] = pd.to_numeric(loaded_df['exam_score'], errors='coerce')
    
    # Merge and clean
    merged = pd.merge(features, loaded_df, on='student_id', how='inner')
    merged = merged.dropna(subset=['exam_score'])
    
    # Prepare features and target
    X = merged.drop(columns=['student_id', 'class_id', 'exam_score'])
    y = merged['exam_score']
    
    # Train-validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def objective(trial):
        param = {
            "verbosity": 0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
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
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        mse = mean_squared_error(y_valid, preds)
        r2 = r2_score(y_valid, preds)
        trial.set_user_attr("mse", mse)
        trial.set_user_attr("r2", r2)
        return -rmse

    # Run optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10, timeout=3600)  # Reduced for local testing

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value (RMSE): {:.4f}".format(-trial.value))
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
        "tree_method": "hist"
    })
    dtrain = xgb.DMatrix(X, label=y)
    final_model = xgb.train(best_params, dtrain, num_boost_round=200)
    
    # Calculate and print final statistics
    preds_all = final_model.predict(dtrain)
    mse_all = mean_squared_error(y, preds_all)
    rmse_all = np.sqrt(mse_all)
    r2_all = r2_score(y, preds_all)
    
    print("\nFinal Model Statistics on All Data:")
    print(f"MSE: {mse_all:.4f}")
    print(f"RMSE: {rmse_all:.4f}")
    print(f"R²: {r2_all:.4f}")
    
    # Save model locally
    joblib.dump(final_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_xgboost()
