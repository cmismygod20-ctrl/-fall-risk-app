# app.py
import streamlit as st
import pandas as pd
from model import train_model, predict

st.set_page_config(page_title="Fall Risk App", layout="centered")

# =========================
# 언어 선택 / 语言选择
# =========================
lang = st.selectbox("🌐 언어 선택 / 语言选择", ["한국어", "中文"])

# =========================
# 제목
# =========================
if lang == "한국어":
    st.title("🧠 낙상 위험 예측 시스템")
    st.write("환자의 정보를 입력하면 낙상 위험을 예측합니다.")
else:
    st.title("🧠 跌倒风险预测系统")
    st.write("输入患者信息，系统将预测跌倒风险。")

# =========================
# 입력 UI
# =========================
age = st.slider("Age / 나이", 40, 100, 65)
balance = st.slider("Balance Score", 0, 100, 50)
gait = st.slider("Gait Speed", 0.1, 2.0, 1.0)
muscle = st.slider("Muscle Strength", 0, 100, 50)
history = st.selectbox("Fall History / 낙상 이력", [0, 1])

# =========================
# 데이터 로드
# =========================
data = pd.read_csv("sample_data.csv")
model = train_model(data)

# =========================
# 예측 버튼
# =========================
if st.button("🔍 예측 / 预测"):

    result, prob = predict(model, {
        "age": age,
        "balance_score": balance,
        "gait_speed": gait,
        "muscle_strength": muscle,
        "history_falls": history
    })

    # 결과 출력
    if lang == "한국어":
        if result == 1:
            st.error(f"⚠️ 고위험 (확률: {prob:.2f})")
        else:
            st.success(f"✅ 저위험 (확률: {1-prob:.2f})")
    else:
        if result == 1:
            st.error(f"⚠️ 高风险 (概率: {prob:.2f})")
        else:
            st.success(f"✅ 低风险 (概率: {1-prob:.2f})")

    # 설명
    if lang == "한국어":
        st.info("이 결과는 머신러닝 모델을 기반으로 한 예측입니다.")
    else:
        st.info("该结果基于机器学习模型预测，仅供参考。")
