import cv2
import numpy as np
import tensorflow as tf

# --- Load TFLite model ---
interpreter = tf.lite.Interpreter(model_path="model/rps_model.tflite")
interpreter.allocate_tensors()

# Lấy thông tin input/output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 📏 In ra kích thước input mà model yêu cầu
print("📐 Model input details:", input_details)
model_input_shape = input_details[0]['shape']  # dạng: [1, H, W, 3]

# Lấy chiều cao và rộng từ model
IMG_SIZE = (model_input_shape[1], model_input_shape[2])  # tự động lấy từ model

# ⚠️ Class names phải trùng với thư mục dataset
CLASSES = ["paper", "rock", "scissors"]

def preprocess(frame):
    """Tiền xử lý ảnh đầu vào để đưa vào model"""
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W, 3)
    return img

def predict_single(frame):
    """Dự đoán class cho một bàn tay"""
    img = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred_idx = np.argmax(output)
    return CLASSES[pred_idx]

def decide_winner(hand1, hand2):
    """
    Luật thắng thua:
    - paper thắng rock
    - rock thắng scissors
    - scissors thắng paper
    """
    if hand1 == hand2:
        return "Draw 🤝"
    if (hand1 == "paper" and hand2 == "rock") or \
       (hand1 == "rock" and hand2 == "scissors") or \
       (hand1 == "scissors" and hand2 == "paper"):
        return "Player 1 wins 🏆"
    else:
        return "Player 2 wins 🏆"

def predict_rps_two_hands(frame_left, frame_right):
    """Dự đoán kết quả giữa 2 tay"""
    hand1 = predict_single(frame_left)
    hand2 = predict_single(frame_right)
    result = decide_winner(hand1, hand2)
    return hand1, hand2, result
