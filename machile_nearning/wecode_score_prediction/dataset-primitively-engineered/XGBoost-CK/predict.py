# Submission ID :       273230
import os
import modal

# Reuse existing image and volume
image = modal.Image.debian_slim().pip_install(
    "xgboost==3.0.2",
    "optuna==4.4.0",
    "scikit-learn==1.7.0",
    "pandas==2.3.0",
    "numpy==2.3.0",
    "joblib==1.5.1"
)

app = modal.App("wecode-score-prediction", image=image)
VOLUME_PATH = "/vol"
volume = modal.Volume.from_name("wecode-prediction-vol", create_if_missing=True)

@app.function(
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    timeout=60*10  # 10 minutes
)
def predict_scores():
    import pandas as pd
    import joblib
    import xgboost as xgb
    
    # Load data from volume
    features_path = os.path.join(VOLUME_PATH, "data", "score_prediction_features.csv")
    ck_path = os.path.join(VOLUME_PATH, "data", "ck-public.csv")
    
    all_features = pd.read_csv(features_path)
    ck_df = pd.read_csv(ck_path).rename(columns={'hash': 'student_id', 'CK': 'exam_score'})
    ck_df['exam_score'] = pd.to_numeric(ck_df['exam_score'], errors='coerce')
    
    # Identify unscored students
    scored_students = set(ck_df['student_id'].dropna())
    all_students = set(all_features['student_id'])
    unscored_students = list(all_students - scored_students)
    
    print(f"Students to predict: {len(unscored_students)}")
    
    # Prepare prediction data
    unscored_data = all_features[all_features['student_id'].isin(unscored_students)]
    X_predict = unscored_data.drop(columns=['student_id', 'class_id', 'exam_score'], errors='ignore')
    
    # Load trained model
    model_path = os.path.join(VOLUME_PATH, "models", "ck_student_score_predictor_optuna_xgb.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)
    
    # Make predictions
    d_predict = xgb.DMatrix(X_predict)
    predictions = model.predict(d_predict)
    
    # Save results
    results = pd.DataFrame({
        'student_id': unscored_data['student_id'],
        'predicted_exam_score': predictions
    })
    output_path = os.path.join(VOLUME_PATH, "predictions", "ck_unscored_predictions.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # Persist to volume
    volume.commit()
    return results.head()

@app.local_entrypoint()
def main():
    # Run prediction
    sample_results = predict_scores.remote()
    print("Prediction completed. Sample results:")
    print(sample_results)
