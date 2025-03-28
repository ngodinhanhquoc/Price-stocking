import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# Load model
model = keras.models.load_model("model_ARIMA.keras")

# Load data
data = pd.read_csv("Final_Nvidia_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Streamlit UI
st.title("📈 Dự đoán giá cổ phiếu NVIDIA (NVDA)")
st.sidebar.header("⚙️ Cài đặt")

# Input parameters
num_days = st.sidebar.slider("Số ngày dự đoán:", 10, 100, 30)
start_date = st.sidebar.text_input("Ngày bắt đầu", "2015/01/01")
end_date = st.sidebar.text_input("Ngày kết thúc", "2025/03/12")

if st.sidebar.button("📊 Dự đoán"):
    try:
        # Lấy dữ liệu trong khoảng thời gian
        df_filtered = data.loc[start_date:end_date]
        df_scaled = scaler.transform(df_filtered)

        # Dự đoán giá
        X_test = np.array([df_scaled[-num_days:]])
        prediction = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(prediction)

        # Vẽ biểu đồ
        fig, ax = plt.subplots()
        ax.plot(df_filtered.index, df_filtered['Close'], label='Giá thực tế', color='blue')
        ax.plot(df_filtered.index[-num_days:], predicted_prices.flatten(), label='Dự đoán', color='red')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Lỗi: {e}")
