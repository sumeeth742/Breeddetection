from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import io
import base64
import cv2
from ultralytics import YOLO


app = FastAPI()

# -----------------------------
# 🧠 Load AI Models
# -----------------------------
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter(model_path="model/breed_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ["Gir", "Jersey", "Murrah", "Sahiwal"]

yolo_model = YOLO("yolov8n.pt")

# -----------------------------
# 📏 CALIBRATION (Adjust if needed)
# -----------------------------
CM_PER_PIXEL = 0.75   # adjust based on your testing

# -----------------------------
# 💾 MongoDB Setup (SECURE)
# -----------------------------

# -----------------------------
# 🐄 Detect Largest Animal
# -----------------------------
def detect_animal(image):
    results = yolo_model(image)

    best_box = None
    max_area = 0

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)

            if area > max_area:
                max_area = area
                best_box = (x1, y1, x2, y2)

    if best_box:
        x1, y1, x2, y2 = best_box
        return image[y1:y2, x1:x2]

    return image

# -----------------------------
# 📏 Measure Animal (Contour)
# -----------------------------
def calculate_measurements_px(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if cv2.contourArea(c) > 1000]

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return float(w), float(h)

    return 0.0, 0.0

# -----------------------------
# 📊 ATC Score
# -----------------------------
def calculate_score(length, height):
    if height == 0:
        return 0

    ratio = length / height

    if ratio > 1.25:
        return 8
    elif ratio > 1.1:
        return 6
    elif ratio > 1.0:
        return 5
    else:
        return 3

# -----------------------------
# 🎨 UI
# -----------------------------
def generate_html(result="", confidence=0, image_data="",
                  length=0, height=0, score=0):

    return f"""
    <html>
    <head>
        <title>Breed Detection AI</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">

        <style>
            body {{
                font-family: Arial;
                background: linear-gradient(135deg, #4CAF50, #2E7D32);
                text-align: center;
                color: white;
            }}

            .container {{
                background: white;
                color: black;
                padding: 25px;
                margin: 50px auto;
                width: 350px;
                border-radius: 15px;
            }}

            button {{
                background: #4CAF50;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 8px;
            }}

            img {{
                width: 100%;
                margin-top: 10px;
                border-radius: 10px;
            }}
        </style>

        <script>
            function previewImage(input) {{
                const file = input.files[0];
                const reader = new FileReader();

                reader.onload = function(e) {{
                    document.getElementById("preview").src = e.target.result;
                    document.getElementById("preview").style.display = "block";
                }}

                reader.readAsDataURL(file);
            }}
        </script>
    </head>

    <body>

        <h2>🐄 Breed Detection AI</h2>

        <div class="container">

            <form action="/predict" method="post" enctype="multipart/form-data">

                <input type="file" name="file" onchange="previewImage(this)" required>

                <img id="preview" src="{image_data}" style="display:{'block' if image_data else 'none'}"/>

                <br><br>
                <button type="submit">Predict</button>

            </form>

            <h3>{result}</h3>
            <p>{confidence:.2f}% Confidence</p>

            <p><b>Length:</b> {length:.2f} cm</p>
            <p><b>Height:</b> {height:.2f} cm</p>
            <p><b>ATC Score:</b> {score}/10</p>

        </div>

    </body>
    </html>
    """

# -----------------------------
# 🌐 Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return generate_html()

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224,224))
    image_np = np.array(image)

    # 🔹 Breed prediction
    img_array = image_np / 255.0
    input_data = np.expand_dims(img_array, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]

    result = labels[np.argmax(output)]
    confidence = float(np.max(output) * 100)

    # 🔹 Detect animal
    animal_img = detect_animal(image_np)

    # 🔹 Measure
    length_px, height_px = calculate_measurements_px(animal_img)

    # 🔹 Convert to cm
    length_cm = length_px * CM_PER_PIXEL
    height_cm = height_px * CM_PER_PIXEL

    # 🔹 Score
    score = calculate_score(length_cm, height_cm)

    

   

    # 🔹 Image display
    image_base64 = base64.b64encode(contents).decode("utf-8")
    image_data = f"data:image/jpeg;base64,{image_base64}"

    return generate_html(result, confidence, image_data, length_cm, height_cm, score)