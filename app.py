from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
file_nvidia_final_path = "Final_Nvidia_data.csv"
df_nvidia_final = pd.read_csv(file_nvidia_final_path)
# Chọn cột giá đóng cửa làm mục tiêu
target = "Close_x"
df_target = df_nvidia_final[target]
def generate_arima_predictions(series, total_length, order=(5,1,0)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=total_length)
    return forecast

# Tạo feature ARIMA cho toàn bộ dữ liệu
arima_forecast = generate_arima_predictions(df_target, len(df_target))
# Định nghĩa lại loss function
custom_objects = {"mse": MeanSquaredError()}
# Load model với custom_objects
model = load_model("model_ARIMA.h5", custom_objects=custom_objects)
manual_test_data = [
    [136.5, 131.1699981689453,131.55999755859375, 197430000, 7.5, 2.34, 5.647198105761642, 97509.03,135.2899932861328]  # Thay bằng dữ liệu thực tế
]
import joblib
scaler = joblib.load("scaler.pkl")
scaled_manual_test = scaler.transform(manual_test_data)  # Chỉ transform!


arima_forecast_value = 135.47655671797887
arima_feature_manual = np.array([arima_forecast_value]).reshape(-1, 1)  # Giá trị ARIMA
final_manual_test = np.concatenate((scaled_manual_test, arima_feature_manual), axis=1)

timesteps = 30  # Số timesteps mô hình yêu cầu
final_manual_test = np.tile(final_manual_test, (1, timesteps, 1))  # Lặp lại dữ liệu

print("New final test shape:", final_manual_test.shape)  # Phải là (1, 30, 10)
prediction = model.predict(final_manual_test)
print(prediction)
# Tạo một dummy array có cùng số feature như scaler đã học
dummy = np.zeros((1, scaler.n_features_in_))  # (1, 9) vì scaler có 9 feature

# Gán giá trị dự đoán vào cột `Close_x` (cột cuối cùng: -1)
dummy[:, -1] = prediction[:, 0]  

# Inverse transform toàn bộ dữ liệu, sau đó lấy lại giá trị `Close_x`
prediction_original = scaler.inverse_transform(dummy)[:, -1]

print("Dự đoán giá trị:", prediction_original)
import numpy as np

# Lấy dữ liệu test của ngày đầu tiên (1 ngày, num_features)
current_input = final_manual_test[:, -1, :].reshape(1, 1, final_manual_test.shape[2])  # (1, 1, num_features)


# Danh sách lưu dự đoán
future_predictions = []

# Lặp 30 lần để dự đoán 30 ngày tiếp theo
for _ in range(30):
    # Dự đoán ngày tiếp theo
    next_prediction = model.predict(current_input)

    # Giả lập scaler để inverse_transform đúng
    dummy = np.zeros((1, scaler.n_features_in_))  # (1, num_features)
    dummy[:, -1] = next_prediction[:, 0]  # Gán giá trị dự đoán vào cột `Close_x`
    next_prediction_original = scaler.inverse_transform(dummy)[:, -1]  # Chuyển về giá trị thực

    # Lưu lại dự đoán gốc
    future_predictions.append(next_prediction_original)

    # Cập nhật dữ liệu input (thay thế bằng ngày vừa dự đoán)
    next_scaled = np.append(current_input[:, :, :-1], np.array([[next_prediction[0]]]), axis=2)
    current_input = next_scaled.reshape(1, 1, next_scaled.shape[2])  # Định dạng lại input

# In kết quả dự đoán 30 ngày tới
print("Dự đoán 30 ngày tiếp theo:", future_predictions)
