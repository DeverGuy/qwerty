# pyrefly: ignore [missing-import]
import cv2
import pickle
import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)

if os.path.exists("embeddings.pkl") and os.path.getsize("embeddings.pkl") > 0:
    with open("embeddings.pkl", "rb") as f:
        database = pickle.load(f)
else:
    database = {}

import threading
import time

# Thread-safe variables
frame_to_process = None
processed_results = []
results_lock = threading.Lock()
frame_lock = threading.Lock()
running = True

# 1. Threaded camera reader to eliminate OpenCV frame buffering lag
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

def worker():
    global frame_to_process, processed_results, running
    while running:
        frame = None
        with frame_lock:
            if frame_to_process is not None:
                frame = frame_to_process.copy()
                frame_to_process = None  # Consume frame
        
        if frame is not None:
            # Scale frame down for faster model inference
            h, w = frame.shape[:2]
            target_width = 640
            scale = target_width / w
            if scale < 1.0:
                target_height = int(h * scale)
                inference_frame = cv2.resize(frame, (target_width, target_height))
            else:
                scale = 1.0
                inference_frame = frame
            
            faces = app.get(inference_frame)
            results = []
            for face in faces:
                emb = face.embedding
                best_name = "Unknown"
                best_distance = 999999
                
                for name, stored_emb in database.items():
                    distance = np.linalg.norm(emb - stored_emb)
                    if distance < best_distance:
                        best_distance = distance
                        best_name = name
                
                if best_distance > 25:
                    best_name = "Unknown"
                
                # Scale face bounding box coordinates back to full resolution
                x1, y1, x2, y2 = face.bbox
                results.append({
                    "bbox": [
                        int(x1 / scale),
                        int(y1 / scale),
                        int(x2 / scale),
                        int(y2 / scale)
                    ],
                    "name": best_name
                })
            
            with results_lock:
                processed_results = results
        else:
            time.sleep(0.01)

# Start inference worker thread
t = threading.Thread(target=worker, daemon=True)
t.start()

# Start camera capture thread
cam = CameraStream(0)

while True:
    ret, raw_frame = cam.read()
    if not ret or raw_frame is None:
        continue
    
    # Mirror the frame horizontally for display and processing
    frame = cv2.flip(raw_frame, 1)
    
    # Pass frame to worker if it is ready to process
    with frame_lock:
        if frame_to_process is None:
            frame_to_process = frame.copy()
            
    # Copy latest results to avoid holding the lock
    with results_lock:
        current_results = list(processed_results)
        
    # Draw original simple UI overlays
    for res in current_results:
        x1, y1, x2, y2 = res["bbox"]
        name = res["name"]
        
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{name}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
    
    cv2.imshow("Club Robot", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

running = False
cam.release()
cv2.destroyAllWindows()