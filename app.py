import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

# =========================
# 🌐 Language system
# =========================
lang = st.selectbox("🌐 Language / 语言选择", ["English", "中文", "한국어"])

texts = {
    "English": {
        "title": "Fall Risk Prediction System",
        "upload": "Upload Image",
        "predict": "Predict",
        "result_high": "⚠️ High Fall Risk",
        "result_low": "✅ Low Fall Risk"
    },
    "中文": {
        "title": "跌倒风险预测系统",
        "upload": "上传图片",
        "predict": "预测",
        "result_high": "⚠️ 高跌倒风险",
        "result_low": "✅ 低跌倒风险"
    },
    "한국어": {
        "title": "낙상 위험 예측 시스템",
        "upload": "이미지 업로드",
        "predict": "예측",
        "result_high": "⚠️ 높은 낙상 위험",
        "result_low": "✅ 낮은 낙상 위험"
    }
}

st.title(texts[lang]["title"])

# =========================
# 📊 Load and Train Model
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("sample_data.csv")

data = load_data()
data.columns = data.columns.str.strip()
X = data[["age", "balance", "gait", "strength", "history"]]
y = data["risk"]

model = LogisticRegression()
model.fit(X, y)

# =========================
# 🎛 User Input
# =========================
age = st.slider("Age", 20, 100, 50)
balance = st.slider("Balance Score", 0, 100, 50)
gait = st.slider("Gait Speed", 0.0, 2.0, 1.0)
strength = st.slider("Muscle Strength", 0, 100, 50)
history = st.selectbox("Fall History", [0, 1, 2, 3])

# =========================
# 🖼 Image Upload + Analysis
# =========================
uploaded_file = st.file_uploader(texts[lang]["upload"], type=["jpg", "png"])

image_factor = 0

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 简单“图像分析”（亮度影响风险）
    img_array = np.array(image)
    brightness = img_array.mean()

    if brightness < 100:
        image_factor = 1  # 增加风险
    else:
        image_factor = 0

# =========================
# 🔮 Prediction
# =========================
if st.button(texts[lang]["predict"]):
    input_data = np.array([[age, balance, gait, strength, history]])

    prediction = model.predict(input_data)[0]

    # 融合图片分析结果
    prediction = max(prediction, image_factor)

    if prediction == 1:
        st.error(texts[lang]["result_high"])
    else:
        st.success(texts[lang]["result_low"])
