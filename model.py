# model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_model(data):
    X = data[['age', 'balance_score', 'gait_speed', 'muscle_strength', 'history_falls']]
    y = data['fall_risk']

    model = LogisticRegression()
    model.fit(X, y)
    return model

def predict(model, input_data):
    df = pd.DataFrame([input_data])
    return model.predict(df)[0], model.predict_proba(df)[0][1]
