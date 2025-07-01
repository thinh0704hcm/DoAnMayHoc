# Submission ID :       273499
import os
import modal

# Define custom image with required dependencies
image = modal.Image.debian_slim().pip_install(
    "xgboost==3.0.2",
    "optuna==4.4.0",
    "scikit-learn==1.7.0",
    "pandas==2.3.0",
    "numpy==2.3.0",
    "joblib==1.5.1"
)

app = modal.App("wecode-score-prediction", image=image)

# Volume configuration
VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("wecode-prediction-vol", create_if_missing=True)

@app.function(
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    timeout=60*60*2  # 2 hours
)
def train_xgboost():
    import pandas as pd
    import numpy as np
    import optuna
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import xgboost as xgb
    import joblib
    
    # Load data from volume - MODIFIED TO USE TBTL SCORES
    features_path = os.path.join(VOLUME_PATH, "data", "score_prediction_features.csv")
    tbtl_path = os.path.join(VOLUME_PATH, "data", "tbtl-public.csv")
    
    features = pd.read_csv(features_path)
    loaded_df = pd.read_csv(tbtl_path)
    loaded_df = loaded_df.rename(columns={'hash': 'student_id', 'accumulated_score': 'exam_score'})  # Changed column mapping
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
            "tree_method": "hist",
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
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        mse = mean_squared_error(y_valid, preds)
        r2 = r2_score(y_valid, preds)
        trial.set_user_attr("mse", mse)
        trial.set_user_attr("r2", r2)
        return -rmse

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500, timeout=3600)  # 500 trials or 1 hour

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
        "tree_method": "gpu_hist",
        "device": "cuda",
        "predictor": "gpu_predictor"
    })
    dtrain = xgb.DMatrix(X, label=y)
    final_model = xgb.train(best_params, dtrain, num_boost_round=200)
    
    # Save model to volume - UPDATED MODEL NAME
    model_path = os.path.join(VOLUME_PATH, "models", "tbtl_student_score_predictor_optuna_xgb.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Commit volume changes
    volume.commit()

@app.local_entrypoint()
def main():
    # Run training
    train_xgboost.remote()
    print("Training started. TensorBoard will be available at the Modal-provided URL.")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating app.")
