# Submission ID :           274275
import os
import shap
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
FEATURES_FILE = "../dataset-student-performance/student_results.csv"
MODEL_PATH = "ck_student_score_predictor_optuna_xgb.pkl"

def train_xgboost():
    # Load student results and QT scores
    student_results = pd.read_csv(FEATURES_FILE)
    
    # Select relevant features
    features = [
        'n_class_problems',
        'hard_solved',
        'total_score_scaled',
        'wecode_score_scaled',
        'submit_count',
        'accept_rate',
        'error_rate',
        'avg_attempt',
        'class_avg_ck',
        'ai_trust_factor',
        'ai_high_accept_low_attempt',
        'ai_rapid_iteration',
        'ai_score_variance',
        'ai_code_complexity'
    ]
    
    # Prepare training data
    train_data = student_results.dropna(subset=['ck_score'])
    X = train_data[features]
    y = train_data['ck_score']
    
    # Train-validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def objective(trial):
        param = {
            "verbosity": 0,
            "objective": "reg:squarederror",
            "tree_method": "hist",  # Changed to GPU method
            "device": "cpu",            # Explicitly use CUDA
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "nthread": -1,  # Use all available threads
        }
        if param["booster"] == "dart":
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        dtrain = xgb.DMatrix(X_train, label=y_train, nthread=-1)  # GPU optimized
        dvalid = xgb.DMatrix(X_valid, label=y_valid, nthread=-1)   # GPU optimized
        
        bst = xgb.train(
            param,
            dtrain,
            num_boost_round=500,  # Increased for better convergence
            evals=[(dvalid, "eval")],
            early_stopping_rounds=50,
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
    study.optimize(objective, n_trials=50, timeout=7200)  # More trials for better tuning

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value (RMSE): {:.4f}".format(-trial.value))
    print("  MSE: {:.4f}".format(trial.user_attrs["mse"]))
    print("  R²: {:.4f}".format(trial.user_attrs["r2"]))
    
    # Train final model with best params
    best_params = trial.params
    best_params.update({
        "verbosity": 0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cpu"
    })
    
    # Use all available data for final model
    dtrain = xgb.DMatrix(X, label=y, nthread=-1)
    final_model = xgb.train(best_params, dtrain, num_boost_round=500)
    
    # Save model
    joblib.dump(final_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # SHAP analysis
    explainer = shap.TreeExplainer(final_model)
    
    # Calculate SHAP values for validation set
    dvalid = xgb.DMatrix(X_valid)
    shap_values = explainer.shap_values(dvalid)
    
    # Global feature importance
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_valid, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_global.png")
    plt.close()
    
    return final_model

if __name__ == "__main__":
    model = train_xgboost()
