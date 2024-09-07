import cv2
import numpy as np
import torch
import pandas as pd
from ultralytics import YOLO
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import logging
import ast
from sort import Sort  # Import the SORT tracker

# disable warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

# cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# YOLOv8 for vehicle detection
vehicle_model = YOLO("yolov8n.pt").to(device)

# Initialize MiniCPM-Llama3-V-2_5-int4 model
model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
model.eval()

# Define detection zone
detection_zone = np.array(["enter the coordinates here"])
# Initialize SORT tracker
sort_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

def is_inside_zone(box, zone):
    x1, y1, x2, y2 = box
    return all(cv2.pointPolygonTest(zone, (x, y), False) >= 0 for x, y in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)])

def preprocess_image(image, target_size=(448, 448)):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to a larger size
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_CUBIC)

    # Apply contrast enhancement
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Apply sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_sharpened = cv2.filter2D(img_enhanced, -1, kernel)

    # Convert to PIL Image
    img_pil = Image.fromarray(img_sharpened)

    return img_pil

prompt = '"""Analyze this car image. Provide the color, make, type, and license plate number of the vehicle. Return the result as a Python dictionary in the format: {"color": "red", "make": "Toyota", "type": "sedan", "license_plate": "ABC123"}. Just return me this dictionary, I dont want any extra text"""'
def analyze_vehicle(image):
    image_pil = preprocess_image(image)
    question = """Analyze this car image. Provide the color, make, type, and license plate number of the vehicle. Return the result as a Python dictionary in the format: {"color": "red", "make": "Toyota", "type": "sedan", "license_plate": "ABC123"}. Just return me this dictionary, I dont want any extra text"""
    msgs = [{'role': 'user', 'content': question}]

    res = model.chat(
        image=image_pil,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7,
    )

    try:
        analysis = ast.literal_eval(res)
    except:
        print(f"Failed to parse LLM response: {res}")
        analysis = {"color": "unknown", "make": "unknown", "type": "unknown", "license_plate": "unknown"}

    return analysis

cap = cv2.VideoCapture(r"inout_video")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

frame_count = 0
vehicle_confidence_threshold = 0.7

vehicle_data = {}
analyzed_vehicles = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [detection_zone], 255)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    input_frame = cv2.resize(masked_frame, (640, 640))
    input_frame = input_frame.transpose((2, 0, 1))
    input_frame = np.ascontiguousarray(input_frame)
    input_frame = torch.from_numpy(input_frame).to(device).float() / 255.0
    input_frame = input_frame.unsqueeze(0)

    with torch.no_grad():
        vehicle_results = vehicle_model(input_frame)

    detections = []
    for vehicle_result in vehicle_results:
        vehicle_boxes = vehicle_result.boxes.cpu().numpy()
        for vehicle_box in vehicle_boxes:
            confidence = vehicle_box.conf[0]
            if confidence >= vehicle_confidence_threshold:
                x1, y1, x2, y2 = map(int, vehicle_box.xyxy[0])

                x1 = int(x1 * width / 640)
                y1 = int(y1 * height / 640)
                x2 = int(x2 * width / 640)
                y2 = int(y2 * height / 640)

                if is_inside_zone([x1, y1, x2, y2], detection_zone):
                    detections.append([x1, y1, x2, y2, confidence])

    # Update SORT tracker
    track_bbs_ids = sort_tracker.update(np.array(detections))

    for track in track_bbs_ids:
        x1, y1, x2, y2, vehicle_id = map(int, track[:5])

        if vehicle_id not in analyzed_vehicles and is_inside_zone([x1, y1, x2, y2], detection_zone):
            # Expand the ROI slightly to capture more context
            expand = 20
            y1_exp = max(0, y1 - expand)
            y2_exp = min(height, y2 + expand)
            x1_exp = max(0, x1 - expand)
            x2_exp = min(width, x2 + expand)

            vehicle_roi = frame[y1_exp:y2_exp, x1_exp:x2_exp]
            analysis = analyze_vehicle(vehicle_roi)

            vehicle_data[vehicle_id] = {
                'frame': frame_count,
                'color': analysis['color'],
                'make': analysis['make'],
                'type': analysis['type'],
                'license_plate': analysis['license_plate']
            }

            analyzed_vehicles.add(vehicle_id)
            print(f"Analyzed vehicle {vehicle_id}: {analysis}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if vehicle_id in vehicle_data:
            cv2.putText(frame, f"ID: {vehicle_id}, Plate: {vehicle_data[vehicle_id]['license_plate']}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.polylines(frame, [detection_zone], True, (255, 0, 0), 2)

    out.write(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

df = pd.DataFrame.from_dict(vehicle_data, orient='index')
df.reset_index(inplace=True)
df.columns = ['vehicle_id', 'frame', 'color', 'make', 'type', 'license_plate']

df = df[~((df['color'] == 'unknown') & (df['make'] == 'unknown') &
          (df['type'] == 'unknown') & (df['license_plate'] == 'unknown'))]

df.to_csv('vehicle_data.csv', index=False)
print(f"Total unique vehicles detected: {len(df)}")
print(df)