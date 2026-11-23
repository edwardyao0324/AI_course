import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import requests
import ollama
import base64
import io
import webbrowser
import threading
import time

from flask import Flask, request, jsonify, render_template_string


MODEL_PATH = "best_model_fold3.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["angry", "happy", "sad", "surprise"]

app = Flask(__name__)

def get_weather():
    try:
        latitude = 24.0518
        longitude = 120.5161

        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={latitude}&longitude={longitude}"
            f"&current=temperature_2m,weather_code"
        )

        response = requests.get(url, timeout=5)
        data = response.json()

        temp = data["current"]["temperature_2m"]
        code = data["current"]["weather_code"]

        weather_map = {
            0: "晴天",
            1: "大致晴朗",
            2: "多雲",
            3: "陰天",
            45: "霧",
            48: "霧",
            51: "毛毛雨",
            61: "小雨",
            63: "中雨",
            65: "大雨",
            71: "小雪",
            80: "陣雨",
            95: "雷雨"
        }

        weather = weather_map.get(code, "未知天氣")
        return weather, temp

    except Exception as e:
        print("天氣取得失敗:", e)
        return "未知", "未知"


class EarlyClassroomCNN(nn.Module):
    def __init__(self):
        super(EarlyClassroomCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = EarlyClassroomCNN().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def get_rudeness_level(emotion):
    mapping = {
        "happy": 0,
        "surprise": 1,
        "sad": 2,
        "angry": 3
    }

    return mapping.get(emotion, 1)


def emotion_to_chinese(emotion):
    mapping = {
        "happy": "開心，狀態穩定",
        "surprise": "驚訝，狀態正常",
        "sad": "難過，狀態偏低落",
        "angry": "生氣，情緒明顯不穩定"
    }

    return mapping.get(emotion, "未知情緒")

def get_weather_level(weather, temp):
    try:
        temp = float(temp)
    except:
        temp = 25

    if weather in ["雷雨", "大雨"]:
        weather_score = 3
    elif weather in ["中雨", "陣雨"]:
        weather_score = 2
    elif weather in ["小雨", "毛毛雨", "陰天", "霧"]:
        weather_score = 1
    else:
        weather_score = 0

    if temp >= 36 or temp <= 5:
        temp_score = 3
    elif temp >= 33 or temp <= 10:
        temp_score = 2
    elif temp >= 30 or temp <= 15:
        temp_score = 1
    else:
        temp_score = 0

    return max(weather_score, temp_score)


def judge_class_level(emotion, weather, temp):
    emotion_level = get_rudeness_level(emotion)
    weather_level = get_weather_level(weather, temp)

    # angry 不應該只判成「可上課但需注意」
    if emotion == "angry":
        if weather_level >= 2:
            level = 3
            decision = "建議停課"
            reason = "學生情緒明顯不穩定，且天氣狀況不佳，容易影響上課安全與品質。"
        else:
            level = 2
            decision = "不太建議上課"
            reason = "雖然天氣狀況尚可，但學生情緒明顯不穩定，建議先安撫後再決定是否上課。"

        return level, decision, reason, emotion_level, weather_level

    total_score = emotion_level + weather_level

    if total_score <= 1:
        level = 0
        decision = "適合上課"
        reason = "學生狀態穩定，天氣也正常。"

    elif total_score <= 2:
        level = 1
        decision = "可上課但需注意"
        reason = "目前仍可上課，但需要稍微注意學生狀態或天氣變化。"

    elif total_score <= 4:
        level = 2
        decision = "不太建議上課"
        reason = "情緒或天氣狀況較差，可能影響上課品質。"

    else:
        level = 3
        decision = "建議停課"
        reason = "目前狀況不佳，建議暫停或改為線上課程。"

    return level, decision, reason, emotion_level, weather_level


def ask_ai(emotion, weather, temp):
    level, decision, reason, emotion_level, weather_level = judge_class_level(
        emotion,
        weather,
        temp
    )

    emotion_zh = emotion_to_chinese(emotion)

    tone_map = {
        "happy": "語氣要平靜、友善。",
        "surprise": "語氣要稍微嚴肅，提醒對方注意。",
        "sad": "語氣要嚴肅、直接，提醒對方先調整狀態。",
        "angry": "語氣要很兇、很直接，但不能罵髒話。"
    }

    tone = tone_map.get(emotion, "語氣自然。")

    prompt = f"""
你是校園上課決策助理。

請根據以下資料，輸出一句上課建議。

學生狀態：{emotion_zh}
天氣：{weather}
溫度：{temp}°C
系統判斷：{decision}
判斷原因：{reason}
語氣要求：{tone}

輸出規則：
- 不可以只提醒注意，必須明確說能不能上課
- 不可以輸出「happy」、「angry」、「sad」、「surprise」
- 不可以輸出「語氣」
- 不可以輸出「學生情緒」
- 不可以列欄位
- 不可以加標題
- 不可以說「好的」
- 有趣一些
- 50字以內
- 必須符合系統判斷：{decision}
"""

    try:
        response = ollama.chat(
            model="qwen2.5:3b",  #要自己下載
            messages=[
                {
                    "role": "system",
                    "content": "你只能輸出一句繁體中文建議，不可輸出情緒英文、語氣標籤或欄位名稱。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "num_ctx": 1024,
                "num_predict": 60,
                "temperature": 0.3
            }
        )

        ai_text = response["message"]["content"].strip()
        ai_text = ai_text.replace("\n", " ").strip()

        bad_prefixes = [
            "angry 語氣：", "happy 語氣：", "sad 語氣：", "surprise 語氣：",
            "angry語氣：", "happy語氣：", "sad語氣：", "surprise語氣：",
            "語氣：", "AI建議：", "建議："
        ]

        for prefix in bad_prefixes:
            if ai_text.startswith(prefix):
                ai_text = ai_text.replace(prefix, "", 1).strip()

        if decision not in ai_text:
            ai_text = f"{decision}，{ai_text}"

        return ai_text

    except Exception as e:
        print("AI 回覆失敗:", e)
        return "AI 未成功啟用，請確認 Ollama 是否正常。"


def detect_face_bbox(pil_image):
    image = np.array(pil_image)

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return {
            "bbox": None
        }

    faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
    x, y, w, h = faces[0]

    return {
        "bbox": {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "image_w": int(image.shape[1]),
            "image_h": int(image.shape[0])
        }
    }

def analyze_image(pil_image):
    image = np.array(pil_image)

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return {
            "emotion": "找不到人臉",
            "weather": "",
            "result": "請確認臉部有在鏡頭畫面中。",
            "bbox": None
        }

    # 取最大的人臉
    faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
    x, y, w, h = faces[0]

    roi = gray[y:y+h, x:x+w]
    face_image = Image.fromarray(roi)

    input_tensor = transform(face_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        emotion = CLASS_NAMES[predicted.item()]

    weather, temp = get_weather()

    level, decision, reason, emotion_level, weather_level = judge_class_level(
        emotion,
        weather,
        temp
    )

    ai_reply = ask_ai(emotion, weather, temp)

    final_result = (
        f"上課判斷：{decision}\n"
        f"原因：{reason}\n\n"
        f"AI建議：{ai_reply}"
    )

    return {
        "emotion": emotion,
        "weather": f"{weather} / {temp}°C",
        "result": final_result,
        "bbox": {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "image_w": int(image.shape[1]),
            "image_h": int(image.shape[0])
        }
    }


HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <title>早八 Survival Guide</title>

    <style>
        body {
            font-family: "Microsoft JhengHei", Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 12px;
            text-align: center;
        }

        h1 {
            margin: 6px 0 10px 0;
            font-size: 26px;
        }

        .video-wrapper {
            position: relative;
            display: inline-block;
            width: 560px;
            max-width: 95vw;
            background: black;
            border: 3px solid #333;
            border-radius: 10px;
            overflow: hidden;
        }

        .video-wrapper video {
            width: 100%;
            display: block;
            background: black;
        }

        #overlayCanvas {
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 10;
        }

        .hint {
            margin-top: 8px;
            font-size: 16px;
            font-weight: bold;
            color: #222;
        }

        .status {
            margin-top: 5px;
            font-size: 14px;
            font-weight: bold;
            color: red;
        }

        .result-container {
            width: 720px;
            max-width: 95vw;
            margin: 14px auto 0 auto;
            text-align: left;
        }

        .box {
            background: white;
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.12);
        }

        .label {
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 5px;
            color: #333;
        }

        .value {
            font-size: 16px;
            white-space: pre-line;
            line-height: 1.45;
            color: #111;
        }

        .loading {
            color: #2563eb;
            font-weight: bold;
        }
    </style>
</head>

<body>
    <h1>早八 Survival Guide</h1>

    <div class="video-wrapper">
        <video id="video" autoplay playsinline muted></video>
        <canvas id="overlayCanvas"></canvas>
    </div>

    <canvas id="canvas" style="display:none;"></canvas>

    <div class="hint">看鏡頭，按下空白鍵進行辨識</div>
    <div id="cameraStatus" class="status">鏡頭啟動中，請允許瀏覽器使用攝影機</div>

    <div class="result-container">
        <div class="box">
            <div class="label">偵測情緒</div>
            <div id="emotion" class="value">尚未辨識</div>
        </div>

        <div class="box">
            <div class="label">今日天氣</div>
            <div id="weather" class="value">尚未辨識</div>
        </div>

        <div class="box">
            <div class="label">AI 助理建議 / 上課判斷結果</div>
            <div id="result" class="value">尚未辨識</div>
        </div>
    </div>

    <script>
        const video = document.getElementById("video");
        const canvas = document.getElementById("canvas");
        const overlayCanvas = document.getElementById("overlayCanvas");
        const cameraStatus = document.getElementById("cameraStatus");

        const emotionBox = document.getElementById("emotion");
        const weatherBox = document.getElementById("weather");
        const resultBox = document.getElementById("result");

        let isAnalyzing = false;
        let isDetectingFace = false;
        let faceDetectTimer = null;
        const FACE_DETECT_INTERVAL = 300;

        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: 640,
                        height: 480
                    },
                    audio: false
                });

                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    video.play();
                    resizeOverlayCanvas();

                    // 鏡頭啟動後，臉框開始持續偵測
                    startFaceDetectLoop();
                };
                cameraStatus.innerText = "鏡頭已啟用，請按空白鍵辨識";
                cameraStatus.style.color = "green";

            } catch (err) {
                cameraStatus.innerText = "無法開啟鏡頭：" + err.message;
                cameraStatus.style.color = "red";
                console.error(err);
            }
        }

        function resizeOverlayCanvas() {
            const rect = video.getBoundingClientRect();
            overlayCanvas.width = Math.round(rect.width);
            overlayCanvas.height = Math.round(rect.height);
        }

        function clearFaceBox() {
            resizeOverlayCanvas();
            const ctx = overlayCanvas.getContext("2d");
            ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        }

        function drawFaceBox(bbox) {
            resizeOverlayCanvas();

            const ctx = overlayCanvas.getContext("2d");
            ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

            if (!bbox) {
                return;
            }

            const scaleX = overlayCanvas.width / bbox.image_w;
            const scaleY = overlayCanvas.height / bbox.image_h;

            const x = bbox.x * scaleX;
            const y = bbox.y * scaleY;
            const w = bbox.w * scaleX;
            const h = bbox.h * scaleY;

            ctx.strokeStyle = "#00ff00";
            ctx.lineWidth = 4;
            ctx.strokeRect(x, y, w, h);

            ctx.fillStyle = "#00ff00";
            ctx.font = "18px Microsoft JhengHei";
            ctx.fillText("Face", x, Math.max(y - 8, 20));
        }
        async function detectFaceOnly() {
            if (isDetectingFace) {
                return;
            }

            if (!video.videoWidth || !video.videoHeight) {
                return;
            }

            isDetectingFace = true;

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            const ctx = canvas.getContext("2d");
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const imageData = canvas.toDataURL("image/jpeg", 0.65);

            try {
                const response = await fetch("/detect_face", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        image: imageData
                    })
                });

                const data = await response.json();
                drawFaceBox(data.bbox);

            } catch (err) {
                console.log("即時臉框偵測失敗:", err);
            } finally {
                isDetectingFace = false;
            }
        }

        function startFaceDetectLoop() {
            if (faceDetectTimer !== null) {
                return;
            }

            faceDetectTimer = setInterval(() => {
                detectFaceOnly();
            }, FACE_DETECT_INTERVAL);
        }
        async function captureAndAnalyze() {
            if (isAnalyzing) {
                return;
            }

            isAnalyzing = true;

            if (!video.videoWidth || !video.videoHeight) {
                isAnalyzing = false;
                alert("鏡頭尚未準備完成，請等畫面出現後再按空白鍵");
                return;
            }

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            const ctx = canvas.getContext("2d");
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            const imageData = canvas.toDataURL("image/jpeg", 0.9);

            emotionBox.innerText = "辨識中...";
            weatherBox.innerText = "取得中...";
            resultBox.innerText = "正在分析情緒、天氣與上課建議...";
            resultBox.classList.add("loading");

            try {
                const response = await fetch("/analyze", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        image: imageData
                    })
                });

                const data = await response.json();

                emotionBox.innerText = data.emotion;
                weatherBox.innerText = data.weather;
                resultBox.innerText = data.result;
                resultBox.classList.remove("loading");

                console.log("bbox:", data.bbox);
                drawFaceBox(data.bbox);

            } catch (err) {
                emotionBox.innerText = "分析失敗";
                weatherBox.innerText = "";
                resultBox.innerText = "錯誤：" + err.message;
                resultBox.classList.remove("loading");
                clearFaceBox();

            } finally {
                isAnalyzing = false;
            }
        }

        document.addEventListener("keydown", function(event) {
            const tag = document.activeElement.tagName;

            if (tag === "INPUT" || tag === "TEXTAREA") {
                return;
            }

            if (event.code === "Space") {
                event.preventDefault();
                captureAndAnalyze();
            }
        });

        window.addEventListener("resize", function() {
            resizeOverlayCanvas();
        });

        startCamera();
    </script>
</body>
</html>
"""
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/analyze", methods=["POST"])
def analyze_api():
    try:
        data = request.get_json()
        image_data = data.get("image", "")

        if not image_data:
            return jsonify({
                "emotion": "沒有圖片",
                "weather": "",
                "result": "沒有收到截圖資料。",
                "bbox": None
            })

        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result = analyze_image(pil_image)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "emotion": "錯誤",
            "weather": "",
            "result": f"後端分析失敗：{str(e)}",
            "bbox": None
        })
        
@app.route("/detect_face", methods=["POST"])
def detect_face_api():
    try:
        data = request.get_json()
        image_data = data.get("image", "")

        if not image_data:
            return jsonify({
                "bbox": None
            })

        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        result = detect_face_bbox(pil_image)

        return jsonify(result)

    except Exception as e:
        print("即時人臉框偵測失敗:", e)
        return jsonify({
            "bbox": None
        })


def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False)