# pyrefly: ignore [missing-import]
import cv2
import pickle
import os
# pyrefly: ignore [missing-import]
import numpy as np

def load_camera_source(config_path="camera_config.txt"):
    # Default source
    source = 0
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                content = f.read().strip()
                if content.isdigit():
                    source = int(content)
                else:
                    source = content
        except Exception:
            pass
            
    print("\n==========================================")
    print("        BIOMETRIC CAMERA SELECTOR         ")
    print("==========================================")
    
    if os.path.exists(config_path):
        print(f"Current Configured Source: {source}")
        choice = input("Press [ENTER] to use this source, or type 'n' to reconfigure: ").strip().lower()
        if choice != 'n':
            return source
            
    print("\nSelect Camera Option:")
    print(" [1] Laptop Built-in Camera")
    print(" [2] Wireless Phone IP Camera (Wi-Fi)")
    print(" [3] Virtual Camera (DroidCam, Iriun, etc.)")
    
    opt = input("Option (1-3) [default: 1]: ").strip()
    if opt == "2":
        url = input("Enter Phone IP URL (e.g., http://192.168.1.50:8080/video): ").strip()
        if not url.startswith("http"):
            url = "http://" + url
        source = url
    elif opt == "3":
        idx = input("Enter camera index (1, 2, etc.) [default: 1]: ").strip()
        source = int(idx) if idx.isdigit() else 1
    else:
        source = 0
        
    try:
        with open(config_path, "w") as f:
            f.write(str(source))
        print(f"Configuration saved to {config_path}\n")
    except Exception as e:
        print(f"Warning: Could not save configuration: {e}")
        
    return source

CAMERA_SOURCE = load_camera_source()

# pyrefly: ignore [missing-import]
import onnxruntime as ort
import threading
import time

# Monkey-patch ONNX Runtime to optimize performance on Windows (limit CPU threads to prevent lag)
original_init = ort.InferenceSession.__init__
def patched_init(self, model_path, sess_options=None, providers=None, provider_options=None, **kwargs):
    if sess_options is None:
        sess_options = ort.SessionOptions()
    # Limit CPU threads to prevent 100% core usage lag
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 2
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Try GPU execution providers first, then CPU
    providers = ['CUDAExecutionProvider', 'DirectMLExecutionProvider', 'CPUExecutionProvider']
    original_init(self, model_path, sess_options, providers, provider_options, **kwargs)
ort.InferenceSession.__init__ = patched_init

# pyrefly: ignore [missing-import]
from insightface.app import FaceAnalysis

# Initialize FaceAnalysis optimized for speed (only detection and recognition)
# We set det_thresh=0.65 to eliminate false detections on background patterns
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.65)

# Load database and auto-normalize embeddings
if os.path.exists("embeddings.pkl") and os.path.getsize("embeddings.pkl") > 0:
    with open("embeddings.pkl", "rb") as f:
        database = pickle.load(f)
    modified = False
    for name, stored_val in database.items():
        if isinstance(stored_val, list):
            normalized_list = []
            for emb in stored_val:
                norm = np.linalg.norm(emb)
                if norm > 0 and not np.isclose(norm, 1.0, atol=1e-3):
                    normalized_list.append(emb / norm)
                    modified = True
                else:
                    normalized_list.append(emb)
            database[name] = normalized_list
        elif isinstance(stored_val, np.ndarray) and stored_val.ndim == 2:
            normalized_rows = []
            for emb in stored_val:
                norm = np.linalg.norm(emb)
                if norm > 0 and not np.isclose(norm, 1.0, atol=1e-3):
                    normalized_rows.append(emb / norm)
                    modified = True
                else:
                    normalized_rows.append(emb)
            database[name] = np.array(normalized_rows)
        else:
            # Single 1D array (legacy format)
            norm = np.linalg.norm(stored_val)
            if norm > 0 and not np.isclose(norm, 1.0, atol=1e-3):
                database[name] = stored_val / norm
                modified = True
    if modified:
        with open("embeddings.pkl", "wb") as f:
            pickle.dump(database, f)
        print("Auto-normalized existing database embeddings.")
else:
    database = {}

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
        if not self.cap.isOpened():
            print(f"\n[ERROR] Could not open camera source: {src}")
            print("If you are using a Wi-Fi IP Camera:")
            print(" 1. Ensure your phone and laptop are connected to the exact same Wi-Fi network.")
            print(" 2. Ensure the IP address and port match what is shown on your phone's app.")
            print(" 3. Try opening the URL in your web browser first to verify the stream is working.\n")
            os._exit(1)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            try:
                ret = self.cap.grab()
                if ret:
                    ret, frame = self.cap.retrieve()
                    if ret and frame is not None:
                        self.ret = ret
                        self.frame = frame
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

# 2. Worker thread for running face inference asynchronously
def worker():
    global frame_to_process, processed_results, running
    while running:
        frame = None
        with frame_lock:
            if frame_to_process is not None:
                frame = frame_to_process.copy()
                frame_to_process = None  # Consume frame
        
        if frame is not None:
            # Scale frame down for faster model inference (320px width)
            h, w = frame.shape[:2]
            target_width = 320
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
                emb_norm = np.linalg.norm(emb)
                emb = emb / emb_norm if emb_norm > 0 else emb
                
                best_name = "Unknown"
                best_distance = 999999
                
                for name, stored_val in database.items():
                    # Handle multi-sample list/2D array and legacy 1D array formats
                    stored_embs = np.atleast_2d(stored_val)
                    distances = np.linalg.norm(stored_embs - emb, axis=1)
                    min_dist = np.min(distances)
                    if min_dist < best_distance:
                        best_distance = min_dist
                        best_name = name
                
                # More strict L2 threshold (1.0) to filter false recognitions
                if best_distance > 1.0:
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
                    "name": best_name,
                    "distance": best_distance
                })
            
            with results_lock:
                processed_results = results
        else:
            time.sleep(0.005)

# Start inference worker thread
t = threading.Thread(target=worker, daemon=True)
t.start()

# Start camera capture thread
cam = CameraStream(CAMERA_SOURCE)

# Cyberpunk HUD UI box drawing helper
def draw_cyber_box(frame, bbox, name, distance=None):
    x1, y1, x2, y2 = bbox
    w_box = x2 - x1
    h_box = y2 - y1
    
    # BGR colors: Emerald Green for registered, Neon Orange/Red for Unknown
    if name != "Unknown":
        color = (80, 255, 100)   # Neon/Emerald Green (BGR)
        status_tag = "SECURED"
    else:
        color = (0, 75, 255)     # Cyber Neon Orange/Red (BGR)
        status_tag = "UNAUTHORIZED"
        
    # 1. Glowing outer box outlines
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 1, lineType=cv2.LINE_AA) # Black drop shadow
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA) # Main colored boundary
    
    # 2. Sleek thick corner brackets
    corner_len = min(22, int(w_box * 0.18), int(h_box * 0.18))
    thickness = 3
    
    # Top-left corner
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness, lineType=cv2.LINE_AA)
    # Top-right corner
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness, lineType=cv2.LINE_AA)
    # Bottom-left corner
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness, lineType=cv2.LINE_AA)
    # Bottom-right corner
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness, lineType=cv2.LINE_AA)
    
    # 3. Sweeping neon scanline
    scan_period = 2.0  # Seconds
    t_cycle = (time.time() % scan_period) / scan_period
    pos = t_cycle * 2 if t_cycle < 0.5 else (1.0 - t_cycle) * 2
    scan_y = int(y1 + pos * h_box)
    
    # Draw scanline with side ticks
    cv2.line(frame, (x1 + 3, scan_y), (x2 - 3, scan_y), color, 1, lineType=cv2.LINE_AA)
    cv2.line(frame, (x1, scan_y - 2), (x1 + 5, scan_y - 2), color, 1, lineType=cv2.LINE_AA)
    cv2.line(frame, (x2 - 5, scan_y - 2), (x2, scan_y - 2), color, 1, lineType=cv2.LINE_AA)
    
    # 4. Futuristic Text Badge
    if name != "Unknown":
        if distance is not None:
            # Distance confidence mapping
            if distance <= 0.4:
                match_pct = int(95 + (0.4 - distance) * 12.5)
            elif distance <= 0.8:
                match_pct = int(75 + (0.8 - distance) / 0.4 * 20)
            elif distance <= 1.0:
                match_pct = int(50 + (1.0 - distance) / 0.2 * 25)
            else:
                match_pct = 0
            label = f"{status_tag} // {name.upper()} // {match_pct}%"
        else:
            label = f"{status_tag} // {name.upper()}"
    else:
        label = f"{status_tag} // EXCLUDE"
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    text_thickness = 1
    
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    
    frame_h, frame_w = frame.shape[:2]
    badge_x1 = x1
    badge_x2 = x1 + text_w + 14
    
    # Shift badge left if it goes off the right edge
    if badge_x2 > frame_w:
        badge_x1 -= (badge_x2 - frame_w)
        badge_x2 = frame_w
        if badge_x1 < 0:
            badge_x1 = 0
            badge_x2 = text_w + 14

    badge_y1 = max(0, y1 - text_h - 10)
    badge_y2 = y1
    # If clipped at the top, move badge below the bounding box
    if badge_y2 - badge_y1 < text_h + 5:
        badge_y1 = y2
        badge_y2 = y2 + text_h + 10
    
    # Draw dark shadow behind badge first
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 0, 0), -1)
    # Draw colored border around badge
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1, lineType=cv2.LINE_AA)
    # Render text inside badge
    cv2.putText(frame, label, (badge_x1 + 7, badge_y2 - 6), font, font_scale, color, text_thickness, lineType=cv2.LINE_AA)
    
    # 5. Central crosshair target dot
    cx, cy = x1 + w_box // 2, y1 + h_box // 2
    cv2.circle(frame, (cx, cy), 2, color, -1)

# Real-time face tracking variables
tracked_faces = {}
next_face_id = 0
LERP_FACTOR = 0.35    # Smoother display box interpolation

last_results = None

# FPS calculation variables
fps_start_time = time.time()
fps_counter = 0
fps = 30.0

cv2.namedWindow("Club Robot", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty("Club Robot", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def show_fullscreen(window_name, img):
    dh, dw = img.shape[:2]
    target_w = int(dh * 16 / 9)
    if target_w > dw:
        canvas = np.zeros((dh, target_w, 3), dtype=np.uint8)
        x_off = (target_w - dw) // 2
        canvas[:, x_off:x_off+dw] = img
        cv2.imshow(window_name, canvas)
    else:
        cv2.imshow(window_name, img)

while True:
    ret, raw_frame = cam.read()
    if not ret or raw_frame is None:
        continue
    
    # Only mirror if it's a local built-in webcam
    if isinstance(CAMERA_SOURCE, int):
        frame = cv2.flip(raw_frame, 1)
    else:
        frame = raw_frame.copy()
        # Rotate 90 degrees clockwise to match portrait phone orientation
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    # Prevent UI from becoming microscopic on high-res phone cameras
    h, w = frame.shape[:2]
    max_h = 720
    if h > max_h:
        scale_d = max_h / h
        frame = cv2.resize(frame, (int(w * scale_d), max_h))
        
    h, w = frame.shape[:2]
    
    # Pass frame to worker if it is ready to process
    with frame_lock:
        if frame_to_process is None:
            frame_to_process = frame.copy()
            
    # Check if there are new processed results from the worker thread
    new_results_available = False
    current_results = []
    with results_lock:
        if processed_results is not last_results:
            current_results = list(processed_results)
            last_results = processed_results
            new_results_available = True

    current_time = time.time()
    
    # 1. Update Tracking with AI Results (when available)
    if new_results_available:
        matched_tracked = set()
        matched_results = set()
        
        potential_matches = []
        for res_idx, res in enumerate(current_results):
            rx1, ry1, rx2, ry2 = res["bbox"]
            rcx, rcy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
            face_size = max(rx2 - rx1, ry2 - ry1)
            
            for fid, f in tracked_faces.items():
                fx1, fy1, fx2, fy2 = f["tracker_bbox"]
                fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                dist = np.sqrt((rcx - fcx)**2 + (rcy - fcy)**2)
                
                # Dynamic threshold: allow fast movement
                if dist < max(350, face_size * 2.5):
                    potential_matches.append((dist, res_idx, fid))
        
        # Sort matches closest first
        potential_matches.sort(key=lambda x: x[0])
        
        # Bipartite matching association
        for dist, res_idx, fid in potential_matches:
            if res_idx not in matched_results and fid not in matched_tracked:
                res = current_results[res_idx]
                tracked_faces[fid]["tracker_bbox"] = list(res["bbox"])
                tracked_faces[fid]["name"] = res["name"]
                tracked_faces[fid]["distance"] = res["distance"]
                tracked_faces[fid]["last_seen"] = current_time
                
                # Crop a new template to track in intermediate frames
                x1, y1, x2, y2 = res["bbox"]
                x1_c = max(0, x1)
                y1_c = max(0, y1)
                x2_c = min(w, x2)
                y2_c = min(h, y2)
                if x2_c > x1_c and y2_c > y1_c:
                    tracked_faces[fid]["template"] = frame[y1_c:y2_c, x1_c:x2_c].copy()
                    
                matched_results.add(res_idx)
                matched_tracked.add(fid)
                
        # Create new tracked faces for unmatched AI detections
        for res_idx, res in enumerate(current_results):
            if res_idx not in matched_results:
                x1, y1, x2, y2 = res["bbox"]
                x1_c = max(0, x1)
                y1_c = max(0, y1)
                x2_c = min(w, x2)
                y2_c = min(h, y2)
                
                new_face = {
                    "bbox": list(res["bbox"]),
                    "tracker_bbox": list(res["bbox"]),
                    "name": res["name"],
                    "distance": res["distance"],
                    "last_seen": current_time
                }
                
                if x2_c > x1_c and y2_c > y1_c:
                    new_face["template"] = frame[y1_c:y2_c, x1_c:x2_c].copy()
                    
                tracked_faces[next_face_id] = new_face
                next_face_id += 1

    # 2. Intermediate Frame Template Tracking (Main Thread - 30 FPS)
    # If the AI is busy, we track the face position using template matching
    for fid, f in tracked_faces.items():
        if "template" in f and f["template"] is not None:
            tx1, ty1, tx2, ty2 = f["tracker_bbox"]
            tw = tx2 - tx1
            th = ty2 - ty1
            
            # Pad the search window to allow for rapid face movement
            pad_w = tw // 2
            pad_h = th // 2
            
            sx1 = max(0, tx1 - pad_w)
            sy1 = max(0, ty1 - pad_h)
            sx2 = min(w, tx2 + pad_w)
            sy2 = min(h, ty2 + pad_h)
            
            # Verify valid dimensions for template matching
            if (sy2 - sy1) > th and (sx2 - sx1) > tw and th > 15 and tw > 15:
                search_region = frame[sy1:sy2, sx1:sx2]
                template_img = f["template"]
                
                try:
                    res = cv2.matchTemplate(search_region, template_img, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    
                    if max_val > 0.65: # Stricter match score validation to prevent background drift
                        dx, dy = max_loc
                        new_x1 = sx1 + dx
                        new_y1 = sy1 + dy
                        new_x2 = new_x1 + tw
                        new_y2 = new_y1 + th
                        
                        f["tracker_bbox"] = [new_x1, new_y1, new_x2, new_y2]
                        
                        # Grab a fresh template from this new position to handle rotation/tilt
                        nx1_c = max(0, new_x1)
                        ny1_c = max(0, new_y1)
                        nx2_c = min(w, new_x2)
                        ny2_c = min(h, new_y2)
                        if nx2_c > nx1_c and ny2_c > ny1_c:
                            f["template"] = frame[ny1_c:ny2_c, nx1_c:nx2_c].copy()
                except Exception:
                    pass

    # 3. Clean up stale tracked faces (timeout if not validated by AI for > 0.6s)
    to_delete = [fid for fid, f in tracked_faces.items() if current_time - f["last_seen"] > 0.6]
    for fid in to_delete:
        del tracked_faces[fid]

    # 4. Interpolate display bboxes (Lerp) to smoothly draw transitions
    for fid, f in tracked_faces.items():
        curr = f["bbox"]
        target = f["tracker_bbox"]
        for i in range(4):
            curr[i] = int(curr[i] + LERP_FACTOR * (target[i] - curr[i]))
        f["bbox"] = curr

    # 5. Draw Cyber HUD Elements
    
    # Glassmorphic telemetry panel (top-left)
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (220, 115), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    
    # Render HUD texts
    cv2.putText(frame, "BIOMETRIC MONITOR", (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(frame, "SYSTEM: ACTIVE", (25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(frame, f"VIDEO FPS: {fps:.1f}", (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(frame, f"TARGETS DETECTED: {len(tracked_faces)}", (25, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(
        frame, 
        "TARGET LOCK: SECURED" if len(tracked_faces) > 0 else "TARGET LOCK: STANDBY", 
        (25, 100), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.38, 
        (80, 255, 100) if len(tracked_faces) > 0 else (120, 120, 120), 
        1, 
        lineType=cv2.LINE_AA
    )

    # Draw tracking boxes for active subjects
    for fid, f in tracked_faces.items():
        draw_cyber_box(frame, f["bbox"], f["name"], f.get("distance"))
    
    show_fullscreen("Club Robot", frame)
    
    # Calculate FPS
    fps_counter += 1
    if time.time() - fps_start_time > 1.0:
        fps = fps_counter / (time.time() - fps_start_time)
        fps_counter = 0
        fps_start_time = time.time()
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

running = False
cam.release()
cv2.destroyAllWindows()