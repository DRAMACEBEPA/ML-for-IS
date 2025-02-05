import os

import time
import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, UDP
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

# === Параметры ===
TRAINING_PERIOD = 0.3 * 60 * 60  # Период сбора обучающей выборки (24 часа)
CAPTURE_INTERFACE = "enp92s0"  # Сетевой интерфейс для захвата трафика
MODEL_PATH = "anomaly_detection_model.h5"  # Путь для сохранения/загрузки модели
SPECIAL_SYMBOLS = "!@#$%^&*()-+"

# === Сбор трафика ===
def capture_traffic(duration, output_file="traffic.csv"):
    """Сбор сетевого трафика и сохранение в CSV."""
    packets = []

    def process_packet(packet):
        if IP in packet:
            packets.append({
                "src_ip": packet[IP].src,
                "dst_ip": packet[IP].dst,
                "protocol": packet[IP].proto,
                "length": len(packet),
                "src_port": packet[TCP].sport if TCP in packet else (packet[UDP].sport if UDP in packet else 0),
                "dst_port": packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else 0)
            })

    print(f"Начало захвата трафика на интерфейсе {CAPTURE_INTERFACE}...")
    sniff(iface=CAPTURE_INTERFACE, prn=process_packet, timeout=duration)
    print(f"Захват завершен. Сохранение данных в {output_file}...")

    # Сохранение в CSV
    df = pd.DataFrame(packets)
    df.to_csv(output_file, index=False)
    print("Данные сохранены.")

# === Оцифровка данных ===
def preprocess_data(input_file):
    """Оцифровка данных и преобразование в матрицу."""
    df = pd.read_csv(input_file)
    df["protocol"] = df["protocol"].astype("category").cat.codes  # Преобразование протоколов в числа
    features = ["protocol", "length", "src_port", "dst_port"]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    return scaled_data

# === Обучение нейронной сети ===
def train_model(training_data, model_path=MODEL_PATH):
    """Обучение нейронной сети на основе обучающей выборки."""
    model = Sequential([
        Dense(64, activation="relu", input_shape=(training_data.shape[1],)),
        Dense(32, activation="relu"),
        Dense(training_data.shape[1], activation="linear")  # Реконструкция входных данных
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])
    model.fit(training_data, training_data, epochs=10, batch_size=32)
    model.save(model_path)
    print(f"Модель сохранена в {model_path}")

# === Обнаружение аномалий ===
def detect_anomalies(test_data, model_path=MODEL_PATH, threshold=0.01):
    """Обнаружение аномалий в тестовых данных."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель {model_path} не найдена. Сначала обучите модель.")
    
    model = load_model(model_path)
    predictions = model.predict(test_data)
    reconstruction_errors = np.mean(np.square(test_data - predictions), axis=1)
    anomalies = np.where(reconstruction_errors > threshold)[0]
    return anomalies, reconstruction_errors

# === Формирование отчетов ===
def generate_report(anomalies, reconstruction_errors, output_file="anomalies_report.txt"):
    """Формирование отчета об аномалиях."""
    with open(output_file, "w") as f:
        if len(anomalies) == 0:
            f.write("Аномалии не обнаружены.\n")
        else:
            f.write("Обнаруженные аномалии:\n")
            for idx in anomalies:
                f.write(f"Индекс: {idx}, Ошибка восстановления: {reconstruction_errors[idx]:.4f}\n")
    print(f"Отчет сохранен в {output_file}")

# === Основной процесс ===
def main():
    # Сбор обучающей выборки
    print("Сбор обучающей выборки...")
    capture_traffic(TRAINING_PERIOD, output_file="training_data.csv")
    training_data = preprocess_data("training_data.csv")

    # Обучение модели
    print("Обучение модели...")
    train_model(training_data)

    # Сбор тестовой выборки
    print("Сбор тестовой выборки...")
    capture_traffic(60, output_file="test_data.csv")  # Захват тестового трафика (например, 1 минута)
    test_data = preprocess_data("test_data.csv")

    # Обнаружение аномалий
    print("Обнаружение аномалий...")
    anomalies, reconstruction_errors = detect_anomalies(test_data)

    # Формирование отчета
    generate_report(anomalies, reconstruction_errors)


if __name__ == "__main__":
    main()
