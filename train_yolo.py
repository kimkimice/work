from ultralytics import YOLO

def train_model():
    # Load the YOLO model
    model = YOLO('yolov8n.pt')  # ใช้โมเดล YOLOv8 nano เป็นฐาน

    # Train the model
    model.train(data='C:/Users/User/PycharmProjects/pythonProject/data.yaml', epochs=100)

    # Save the model
    model.save('best.pt')  # บันทึกโมเดลที่ถูกเทรนแล้ว

if __name__ == "__main__":
    train_model()
