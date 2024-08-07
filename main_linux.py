import time
import os
import cv2
import queue
import threading
import numpy as np
from datetime import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction
from sshkeyboard import listen_keyboard, stop_listening
import mysql.connector
from mysql.connector import Error

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--stream", type=str, help="RTSP address of video stream.")
parser.add_argument('--monitor', default=False, action=BooleanOptionalAction,
                    help="View the live stream. If no monitor is connected then leave this disabled (no Raspberry Pi SSH sessions).")
parser.add_argument("--yolo", type=str,
                    help="Enables YOLO object detection. Enter a comma separated list of objects you'd like the program to record. The list can be found in the coco.names file")
parser.add_argument("--model", default='yolov8n', type=str,
                    help="Specify which model size you want to run. Default is the nano model.")
args = vars(parser.parse_args())

rtsp_stream = args["stream"]
monitor = args["monitor"]
if args["yolo"]:
    yolo_list = [s.strip() for s in args["yolo"].split(",")]
    yolo_on = True
else:
    yolo_on = False

# Set up variables for YOLO detection
if yolo_on:
    from ultralytics import YOLO

    stop_error = False

    CONFIDENCE = 0.5
    font_scale = 1
    thickness = 1
    labels = open("coco.names").read().strip().split("\n")
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    model = YOLO(args["model"] + ".pt")

    # Check if the user provided list has valid objects
    for coconame in yolo_list:
        if coconame not in labels:
            print("Error! '" + coconame + "' not found in coco.names")
            stop_error = True
    if stop_error:
        exit("Exiting")

# Set up other internal variables
loop = True
cap = cv2.VideoCapture(rtsp_stream)
fps = cap.get(cv2.CAP_PROP_FPS)
period = 1 / fps
recording = False
ret, img = cap.read()
if img.shape[1] / img.shape[0] > 1.55:
    res = (256, 144)
else:
    res = (216, 162)
blank = np.zeros((res[1], res[0]), np.uint8)
resized_frame = cv2.resize(img, res)
gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
old_frame = np.float32(cv2.GaussianBlur(gray_frame, (5, 5), 0))
if monitor:
    cv2.namedWindow(rtsp_stream, cv2.WINDOW_NORMAL)

q = queue.Queue(maxsize=10)

# Thread for receiving the stream's frames so they can be processed
# If camera disconnects it will automatically try to reconnect every 5 seconds
def receive_frames():
    global cap
    if cap.isOpened():
        ret, frame = cap.read()
        while loop:
            ret, frame = cap.read()
            if ret:
                if not q.full():
                    q.put(frame)
            else:
                now_time = datetime.now().strftime('%H-%M-%S')
                print(now_time + " Camera disconnected. Attempting to reconnect.")
                while loop:
                    cap = cv2.VideoCapture(rtsp_stream)
                    if cap.isOpened():
                        now_time = datetime.now().strftime('%H-%M-%S')
                        print(now_time + " Camera successfully reconnected.")
                        break
                    else:
                        time.sleep(5)

# Functions for detecting key presses
def press(key):
    global loop
    if key == 'q':
        loop = False

def input_keyboard():
    listen_keyboard(
        on_press=press,
    )

# Process YOLO object detection
def process_yolo():
    global img

    results = model.predict(img, conf=CONFIDENCE, verbose=False)[0]
    object_found = False
    detected_object = None

    # Loop over the detections
    for data in results.boxes.data.tolist():
        # Get the bounding box coordinates, confidence, and class id
        xmin, ymin, xmax, ymax, confidence, class_id = data

        # Converting the coordinates and the class id to integers
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        class_id = int(class_id)

        if labels[class_id] in yolo_list:
            object_found = True
            detected_object = labels[class_id]

            # Print detection information based on object type
            print(f"Detected {detected_object} with confidence {confidence:.2f}")

            # Draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors[class_id]]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)
            text = f"{detected_object}: {confidence:.2f}"
            # Calculate text width & height to draw the transparent boxes as background of the text
            (text_width, text_height) = \
            cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = xmin
            text_offset_y = ymin - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = img.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            # Add opacity (transparency to the box)
            img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
            # Now put the text (label: confidence %)
            cv2.putText(img, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(0, 0, 0),
                        thickness=thickness)

    return object_found, detected_object

# Function to save images and save to database
def save_image(image, detected_object):
    folder_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_path = os.path.join("/home/YOUR_USERNAME/PycharmProjects/pythonProject/recordings", detected_object, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.join(folder_path, f"{detected_object}_{folder_name}.jpg")
    # Create the URL
    full_path = f"http://YOUR_SERVER_IP/recordings/{detected_object}/{folder_name}/{detected_object}_{folder_name}.jpg"
    cv2.imwrite(filename, image)
    print(f"Saved image: {filename}")
    save_to_database(full_path, detected_object)  # บันทึกข้อมูลลงในฐานข้อมูล

# ฟังก์ชันสำหรับเชื่อมต่อและบันทึกข้อมูลลงในฐานข้อมูล
def save_to_database(file_path, detected_object):
    try:
        connection = mysql.connector.connect(
            host='YOUR_SERVER_IP',  # ใส่ IP ของเซิร์ฟเวอร์
            database='yolo_database',  # ใส่ชื่อฐานข้อมูลของคุณ
            user='root',  # ใส่ชื่อผู้ใช้ฐานข้อมูลของคุณ
            password=''  # ใส่รหัสผ่านฐานข้อมูลของคุณ (ถ้ามี)
        )
        if connection.is_connected():
            cursor = connection.cursor()
            query = """INSERT INTO detections (file_path, detected_object) VALUES (%s, %s)"""
            record = (file_path, detected_object)
            cursor.execute(query, record)
            connection.commit()
            print(f"Record inserted successfully into detections table, file path: {file_path}, detected object: {detected_object}")

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Start the background threads
receive_thread = threading.Thread(target=receive_frames)
receive_thread.start()
keyboard_thread = threading.Thread(target=input_keyboard)
keyboard_thread.start()

# Main loop
while loop:
    if not q.empty():
        img = q.get()

        # Resize image, make it grayscale, then blur it
        resized_frame = cv2.resize(img, res)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        final_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Ensure the size of old_frame and final_frame match
        if old_frame.shape != final_frame.shape:
            old_frame = np.float32(final_frame)

        # Use accumulateWeighted to reduce noise and get smoother frames
        cv2.accumulateWeighted(final_frame, old_frame, 0.5)
        result_frame = cv2.convertScaleAbs(old_frame)

        # Calculate difference between current and previous frame
        diff = cv2.absdiff(final_frame, result_frame)
        result = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)[1]

        # Count the number of frames where the motion value exceeds the threshold value.
        if np.sum(result) > 350:
            if yolo_on:
                object_found, detected_object = process_yolo()
                if object_found:
                    # Save image in a new thread
                    threading.Thread(target=save_image, args=(img, detected_object)).start()

        # Monitor the stream
        if monitor:
            cv2.imshow(rtsp_stream, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                loop = False
    else:
        time.sleep(period / 2)

# Gracefully end threads and exit
stop_listening()
receive_thread.join()
keyboard_thread.join()
cv2.destroyAllWindows()
print("Exiting")
