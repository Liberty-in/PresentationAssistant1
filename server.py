import pandas as pd
import numpy as np

import cv2
import json
import torch
import uvicorn
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as F
import asyncio

import base64
from fastapi import FastAPI, WebSocket

dict = {
    0: "two close fingers up",
    1: "ok",
    2: "cool",
    3: "hand up",
    4: "two fingers up",
    5: "heart",
    6: "zig",
    7: "pity",
    8: "two hand"
}

model1 = models.detection.ssdlite320_mobilenet_v3_large(weights='DEFAULT')
model1.load_state_dict(torch.load("model_state_dict2.pth", weights_only=False))
model1.eval()

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    i = 0
    c = 0
    pred = False
    print("Новое подключение")
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            np_array = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            i += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = F.to_tensor(image)
            input_image = image.unsqueeze(0)
            with torch.no_grad():
                predictions = model1(input_image)
            for j in range(len(predictions[0]['boxes'])):
                box = predictions[0]['boxes'][j].numpy()
                score = predictions[0]['scores'][j].item()
                label = predictions[0]['labels'][j].item()
                if score > 0.5:
                    n = box
                    cv2.rectangle(frame, (int(n[0]), int(n[1])), (int(n[2]), int(n[3])), color=(0, 255, 0), thickness=2)
                    cv2.putText(frame,
                                dict[label - 1],
                                (int(n[2]), int(n[3])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1)

                    if pred == False:
                        pred = True
                        info = predictions[0]['labels'][j].item()
                        #await websocket.send(info)
                    c += 1
            if pred == True and c == 0:
                pred = False

            _, jpeg_frame = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg_frame.tobytes()
            info = base64.b64encode(frame_bytes).decode('utf-8')
            await websocket.send_text(info)

    except Exception as e:
        print(f"Ошибка: {str(e)}")
    finally:
        print("Подключение закрыто")

@app.get("/get_doc")
async def root():
    df = pd.read_csv("DB.csv")
    data = df.to_dict(orient="records")
    return {"data": data}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)