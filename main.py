import cv2
from logic import predict_rps_two_hands

# Nhập URL IP camera của điện thoại tại đây:
# Mở app IP Webcam → lấy URL dạng: http://192.168.xx.xx:8080/video
ip_url = "http://192.168.1.58:4747/video"  # ĐỔI địa chỉ này cho phù hợp

# Nếu muốn dùng webcam laptop thay vì IP camera → dùng:
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(ip_url)

if not cap.isOpened():
    print("Không thể kết nối camera. Kiểm tra lại URL hoặc kết nối Wi-Fi.")
    exit()

print("Camera kết nối thành công. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được khung hình từ camera.")
        break

    # Chia đôi khung hình: trái = Player 1, phải = Player 2
    h, w, _ = frame.shape
    frame_left = frame[:, :w//2]     # Player 1
    frame_right = frame[:, w//2:]    # Player 2

    # Dự đoán
    hand1, hand2, result = predict_rps_two_hands(frame_left, frame_right)

    # Hiển thị thông tin lên khung hình
    cv2.putText(frame, f"P1: {hand1}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"P2: {hand2}", (w//2 + 30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"Result: {result}", (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Rock-Paper-Scissors", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
