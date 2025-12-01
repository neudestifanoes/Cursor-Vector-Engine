from typing import List


import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from features import extract_ssvep_features

# ----------------- FastAPI app -----------------
app = FastAPI(
    title="SSVEP Backend",
    description="Neural cursor control backend API",
    version="0.1.0",
)

# CORS: allow frontend on localhost:3000 to talk to backend:8000
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,     # during dev you could also use ["*"]
    allow_credentials=True,
    allow_methods=["*"],       # very important: allow OPTIONS
    allow_headers=["*"],
)

# ----------------- WebSocket manager -----------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, message: dict):
        # Send JSON to all connected clients
        disconnected = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        # clean up dead connections
        for ws in disconnected:
            self.disconnect(ws)


manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for live prediction updates.

    - Frontend connects here: ws://127.0.0.1:8000/ws
    - Whenever /predict is called, we broadcast the result to all clients.
    """
    await manager.connect(websocket)
    try:
        # optional: receive messages from client (for now we just ignore or log)
        while True:
            _ = await websocket.receive_text()  # keeps connection alive
            # you could also send pings or acks here if you want
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ----------------- Models & schemas -----------------
class PredictRequest(BaseModel):
    data: list[list[float]]
    model_name: str = "lda"


# Load models once at startup
lda_model = joblib.load("ssvep_lda_model.joblib")
svm_model = joblib.load("ssvep_svm_model.joblib")
models = {"lda": lda_model, "svm": svm_model}
print(f"Models loaded. LDA: {lda_model is not None} | SVM: {svm_model is not None}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "lda_loaded": "lda" in models,
        "svm_loaded": "svm" in models,
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    # req.data is channels x samples
    arr = np.array(req.data, dtype=float)
    if arr.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail="data must be a 2D array: channels x samples",
        )

    # Add batch dimension: (1, channels, samples)
    X = arr[np.newaxis, :, :]

    # Extract SSVEP features using the same function as training
    X_feat = extract_ssvep_features(X)  # shape: (1, n_features)

    model = models.get(req.model_name)
    if model is None:
        raise HTTPException(status_code=400, detail="Unknown model_name")

    # Predict probabilities
    probs = model.predict_proba(X_feat)[0]
    classes = model.classes_

    class_probs = {str(cls): float(p) for cls, p in zip(classes, probs)}
    pred_idx = int(np.argmax(probs))
    pred_label = str(classes[pred_idx])

    response = {
        "model_name": req.model_name,
        "predicted_label": pred_label,
        "class_probabilities": class_probs,
    }

    # 🚀 Broadcast this prediction to all WebSocket clients
    await manager.broadcast_json({
        "type": "prediction",
        **response,
    })

    return response

