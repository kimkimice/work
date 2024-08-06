from ultralytics import YOLO


def check_trained_model_classes(model_path):
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Extract the class names from the model
    class_names = model.names

    # Print the class names
    print("Classes in the trained model:")
    for idx, name in enumerate(class_names):
        print(f"Class {idx}: {name}")


if __name__ == "__main__":
    model_path = 'C:/Users/User/PycharmProjects/facere/best.pt'
    check_trained_model_classes(model_path)
