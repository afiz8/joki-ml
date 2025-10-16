# ml_ml_predictor.py
"""
Contoh pipeline ML untuk prediksi hasil match Mobile Legends (win/lose).
File ini membuat dataset sintetis bila tidak ada data asli, lalu:
 - preprocessing (encoding, scaling)
 - training (RandomForest + LogisticRegression baseline)
 - evaluasi (accuracy, f1, confusion matrix)
 - simpan model
 - contoh prediksi
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import os

RANDOM_STATE = 42
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def make_synthetic_data(n_samples=2000, random_state=RANDOM_STATE):
    """Membuat dataset contoh mirip match statistics."""
    rng = np.random.RandomState(random_state)
    # fitur numerik
    gold_diff = rng.normal(loc=0, scale=2000, size=n_samples).astype(int)       # team gold - enemy gold
    kills_diff = rng.normal(loc=0, scale=6, size=n_samples).astype(int)        # team kills - enemy kills
    towers_diff = rng.normal(loc=0, scale=2, size=n_samples).astype(int)       # towers destroyed diff
    avg_lvl_diff = rng.normal(loc=0, scale=1.5, size=n_samples).round(1)       # avg level diff

    # fitur kategori (role/hero composition simplified)
    # contoh beberapa hero-role kombinasi: tank, marksman, mage, assassin, fighter, support
    compositions = ['tank,marksman,mage,assassin,fighter',
                    'tank,marksman,marksman,mage,support',
                    'tank,fighter,assassin,marksman,mage',
                    'marksman,mage,assassin,fighter,support',
                    'tank,support,marksman,mage,fighter']
    comp = rng.choice(compositions, size=n_samples, p=[0.2,0.2,0.25,0.2,0.15])

    # target: win (1) or lose (0) with probabilistic rule using features
    # buat aturan sederhana: jika gold_diff + 200*kills_diff + 1000*towers_diff > threshold => lebih besar peluang menang
    score = gold_diff + 200 * kills_diff + 1000 * towers_diff + (avg_lvl_diff * 300)
    prob_win = 1 / (1 + np.exp(-score / 3000))  # sigmoidal mapping
    win = (rng.rand(n_samples) < prob_win).astype(int)

    df = pd.DataFrame({
        'gold_diff': gold_diff,
        'kills_diff': kills_diff,
        'towers_diff': towers_diff,
        'avg_lvl_diff': avg_lvl_diff,
        'composition': comp,
        'win': win
    })
    return df

def load_or_create_data(path=None):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded data from {path}, shape = {df.shape}")
    else:
        print("No data file found — creating synthetic dataset.")
        df = make_synthetic_data(4000)
    return df

def build_and_train(df, save_model=True):
    # fitur dan target
    X = df.drop(columns=['win'])
    y = df['win']

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)

    # preprocessing:
    numeric_features = ['gold_diff', 'kills_diff', 'towers_diff', 'avg_lvl_diff']
    categorical_features = ['composition']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # model pipeline (RandomForest)
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=200))
    ])

    # train
    rf_pipeline.fit(X_train, y_train)

    # eval
    y_pred = rf_pipeline.predict(X_test)
    print("RandomForest evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 score:", f1_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # simple baseline logistic regression
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    print("LogisticRegression evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred_lr))
    print("F1 score:", f1_score(y_test, y_pred_lr))

    if save_model:
        joblib.dump(rf_pipeline, os.path.join(MODEL_DIR, 'rf_mlpredictor.joblib'))
        joblib.dump(lr_pipeline, os.path.join(MODEL_DIR, 'lr_mlpredictor.joblib'))
        print(f"Saved models to {MODEL_DIR}/")

    return rf_pipeline, lr_pipeline, (X_test, y_test)

def predict_example(pipeline, example):
    """
    example: dict berisi fitur sesuai X dataframe, mis:
    {
      'gold_diff': 1500,
      'kills_diff': 3,
      'towers_diff': 1,
      'avg_lvl_diff': 0.7,
      'composition': 'tank,marksman,mage,assassin,fighter'
    }
    """
    df_ex = pd.DataFrame([example])
    prob = pipeline.predict_proba(df_ex)[0,1] if hasattr(pipeline, 'predict_proba') else None
    pred = pipeline.predict(df_ex)[0]
    return pred, prob

if __name__ == "__main__":
    # 1) load data (ganti path_csv jika lo punya data nyata)
    df = load_or_create_data(path=None)

    # 2) build + train
    rf_pipe, lr_pipe, (X_test, y_test) = build_and_train(df, save_model=True)

    # 3) contoh prediksi
    example = {
      'gold_diff': 1200,
      'kills_diff': 2,
      'towers_diff': 1,
      'avg_lvl_diff': 0.5,
      'composition': 'tank,marksman,mage,assassin,fighter'
    }
    pred, prob = predict_example(rf_pipe, example)
    print("Example pred -> pred:", pred, "prob_win:", prob)
