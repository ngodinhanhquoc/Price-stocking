import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load scaler và mô hình
scaler = joblib.load("scaler.pkl")
custom_objects = {"mse": MeanSquaredError()}
model = load_model("model_ARIMA.h5", custom_objects=custom_objects)

def generate_arima_predictions(series, total_length, order=(5,1,0)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=total_length)
    return forecast

st.title("Dự đoán giá trị bằng ARIMA và Mô hình học sâu")

uploaded_file = st.file_uploader("Tải lên dữ liệu CSV", type=["csv"])
if uploaded_file is not None:
    df_nvidia_final = pd.read_csv(uploaded_file)
    target = "Close_x"
    df_target = df_nvidia_final[target]
    
    # ARIMA Feature
    arima_forecast = generate_arima_predictions(df_target, len(df_target))
    
    st.write("Dữ liệu tải lên:")
    st.dataframe(df_nvidia_final.head())
    
    manual_test_data = [[136.5, 131.17, 131.56, 197430000, 7.5, 2.34, 5.64, 97509.03, 135.28]]
    scaled_manual_test = scaler.transform(manual_test_data)
    
    arima_forecast_value = arima_forecast.iloc[-1]
    arima_feature_manual = np.array([arima_forecast_value]).reshape(-1, 1)
    final_manual_test = np.concatenate((scaled_manual_test, arima_feature_manual), axis=1)
    final_manual_test = np.tile(final_manual_test, (1, 30, 1))
    
    prediction = model.predict(final_manual_test)
    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[:, -1] = prediction[:, 0]
    prediction_original = scaler.inverse_transform(dummy)[:, -1]
    
    st.write("Dự đoán giá trị:", prediction_original[0])
    
    # Dự đoán 30 ngày tiếp theo
    current_input = final_manual_test[:, -1, :].reshape(1, 1, final_manual_test.shape[2])
    future_predictions = []
    
    for _ in range(30):
        next_prediction = model.predict(current_input)
        dummy = np.zeros((1, scaler.n_features_in_))
        dummy[:, -1] = next_prediction[:, 0]
        next_prediction_original = scaler.inverse_transform(dummy)[:, -1]
        future_predictions.append(next_prediction_original[0])
        next_scaled = np.append(current_input[:, :, :-1], np.array([[next_prediction[0]]]), axis=2)
        current_input = next_scaled.reshape(1, 1, next_scaled.shape[2])
    
    # Vẽ biểu đồ
    st.write("Dự đoán 30 ngày tiếp theo:")
    fig, ax = plt.subplots()
    ax.plot(range(1, 31), future_predictions, marker='o', linestyle='-')
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá dự đoán")
    ax.set_title("Dự đoán giá trong 30 ngày tới")
    st.pyplot(fig)
