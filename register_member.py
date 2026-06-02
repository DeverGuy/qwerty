# pyrefly: ignore [missing-import]
import cv2
import pickle
import os
import time
import threading
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

# Monkey-patch ONNX Runtime to optimize performance on Windows (limit CPU threads to prevent lag)
# pyrefly: ignore [missing-import]
import onnxruntime as ort
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

# Initialize FaceAnalysis optimized for speed (detection and recognition only)
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.65)

# Load existing database
if os.path.exists("embeddings.pkl") and os.path.getsize("embeddings.pkl") > 0:
    with open("embeddings.pkl", "rb") as f:
        database = pickle.load(f)
else:
    database = {}

name = input("Enter member name: ").strip()
if not name:
    print("Name cannot be empty.")
    exit()

# Case-insensitive overwrite check
existing_name = None
for k in database.keys():
    if k.lower() == name.lower():
        existing_name = k
        break

if existing_name:
    ans = input(f"Member '{existing_name}' already exists. Overwrite? (y/n): ").strip().lower()
    if ans != 'y':
        print("Registration cancelled.")
        exit()
    name = existing_name  # Preserve original key casing

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

cam = CameraStream(CAMERA_SOURCE)

# Registration states
STATE_ALIGNMENT = 0
STATE_CAPTURING = 1
STATE_COMPLETED = 2

state = STATE_ALIGNMENT
samples = []
max_samples = 30
success_timer = None

def draw_cyber_corners(frame, bbox, color, thickness=3, corner_len=18):
    x1, y1, x2, y2 = bbox
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

print("\n--- Starting Camera Window ---")
print("Align your face in the camera stream and press [SPACE] to begin registration.")
print("Press 'q' to exit fullscreen.")

cv2.namedWindow("Register Member", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty("Register Member", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

    # Downscale frame for fast inference
    target_width = 320
    scale = target_width / w
    if scale < 1.0:
        target_height = int(h * scale)
        inference_frame = cv2.resize(frame, (target_width, target_height))
    else:
        scale = 1.0
        inference_frame = frame

    # Run face detection
    faces = app.get(inference_frame)

    if state == STATE_ALIGNMENT:
        # Glassmorphic top header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        cv2.putText(
            frame, 
            f"BIOMETRIC ENROLLMENT // SUBJECT: {name.upper()}", 
            (15, 26), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 255), 
            1, 
            lineType=cv2.LINE_AA
        )

        # Glassmorphic instruction footer
        overlay_foot = frame.copy()
        cv2.rectangle(overlay_foot, (0, h - 50), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay_foot, 0.5, frame, 0.5, 0, frame)
        
        cv2.putText(
            frame,
            "ALIGN FACE & PRESS [SPACE] TO START CAPTURE",
            (15, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA
        )

        # Draw a beautiful circular scanning reticle in the center
        cx, cy = w // 2, h // 2
        cv2.circle(frame, (cx, cy), 90, (120, 120, 120), 1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 95, (80, 80, 80), 1, lineType=cv2.LINE_AA)
        
        # Scanner ticks
        tick_len = 15
        cv2.line(frame, (cx - 110, cy), (cx - 110 + tick_len, cy), (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(frame, (cx + 110 - tick_len, cy), (cx + 110, cy), (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 110), (cx, cy - 110 + tick_len), (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(frame, (cx, cy + 110 - tick_len), (cx, cy + 110), (0, 255, 255), 2, lineType=cv2.LINE_AA)

        # Draw scanning lines for detected faces
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)
            # Neon Cyan box and corners
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1, lineType=cv2.LINE_AA)
            draw_cyber_corners(frame, (x1, y1, x2, y2), (255, 255, 0), thickness=2)
            cv2.putText(frame, "READY", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, lineType=cv2.LINE_AA)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if len(faces) > 0:
                state = STATE_CAPTURING
                print("Acquisition started...")
            else:
                print("No face detected! Cannot start acquisition.")
        elif key == ord('q'):
            break

    elif state == STATE_CAPTURING:
        # Glassmorphic top header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(
            frame, 
            f"BIOMETRIC ENROLLMENT // SUBJECT: {name.upper()}", 
            (15, 26), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 255), 
            1, 
            lineType=cv2.LINE_AA
        )

        # Dynamic instructions based on progress
        count = len(samples)
        if count < 10:
            prompt = "LOOK STRAIGHT AT THE CAMERA"
            color_state = (255, 255, 0) # Cyan
        elif count < 20:
            prompt = "SLOWLY ROTATE HEAD LEFT & RIGHT"
            color_state = (0, 255, 255) # Yellow
        else:
            prompt = "SLOWLY TILT HEAD UP & DOWN"
            color_state = (255, 100, 255) # Pink/Violet

        # Glassmorphic footer panel
        overlay_foot = frame.copy()
        cv2.rectangle(overlay_foot, (0, h - 60), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay_foot, 0.5, frame, 0.5, 0, frame)

        cv2.putText(
            frame,
            f"INSTRUCTION: {prompt}",
            (15, h - 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color_state,
            1,
            lineType=cv2.LINE_AA
        )

        # Render neon-green progress bar
        bar_width = int((count / max_samples) * (w - 30))
        cv2.rectangle(frame, (15, h - 22), (w - 15, h - 10), (45, 45, 45), -1) # Background bar
        cv2.rectangle(frame, (15, h - 22), (15 + bar_width, h - 10), (80, 255, 100), -1) # Glowing progress
        cv2.putText(
            frame,
            f"ACQUISITION PROGRESS: {count}/{max_samples}",
            (15, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (80, 255, 100),
            1,
            lineType=cv2.LINE_AA
        )

        # Look for a high-confidence face to record
        recorded_this_frame = False
        for face in faces:
            if face.det_score > 0.75:
                x1, y1, x2, y2 = face.bbox
                x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)
                w_box = x2 - x1
                h_box = y2 - y1
                
                # Check face size (ensure it's not a tiny background face)
                if w_box > 80:
                    emb = face.embedding
                    norm = np.linalg.norm(emb)
                    normalized_emb = emb / norm if norm > 0 else emb
                    samples.append(normalized_emb)
                    recorded_this_frame = True
                    
                    # Double-border box + corners in Neon Green
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 100), 1, lineType=cv2.LINE_AA)
                    draw_cyber_corners(frame, (x1, y1, x2, y2), (80, 255, 100), thickness=2)
                    
                    # Sweeping scanline inside active capture box
                    scan_period = 1.0
                    t_cycle = (time.time() % scan_period) / scan_period
                    pos = t_cycle * 2 if t_cycle < 0.5 else (1.0 - t_cycle) * 2
                    scan_y = int(y1 + pos * h_box)
                    cv2.line(frame, (x1, scan_y), (x2, scan_y), (80, 255, 100), 1, lineType=cv2.LINE_AA)
                    
                    cv2.putText(frame, "RECORDING", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 255, 100), 1, lineType=cv2.LINE_AA)
                    break

        if not recorded_this_frame:
            # Alert user if no high-quality face is found
            cv2.putText(
                frame,
                "ALIGN YOUR FACE CLEARLY",
                (w // 2 - 110, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 75, 255),
                1,
                lineType=cv2.LINE_AA
            )

        if len(samples) >= max_samples:
            state = STATE_COMPLETED
            success_timer = time.time()
            # Save to database
            database[name] = samples
            with open("embeddings.pkl", "wb") as f:
                pickle.dump(database, f)
            print(f"Successfully trained model with {max_samples} samples for '{name}'.")

        key = cv2.waitKey(100) & 0xFF # 100ms spacing between captures
        if key == ord('q'):
            break

    elif state == STATE_COMPLETED:
        # Full overlay message
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 20, 0), -1) # Dark green tint
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(
            frame,
            "REGISTRATION COMPLETE!",
            (w // 2 - 150, h // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 255, 100),
            2,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"BIOMETRIC TEMPLATE FOR {name.upper()} CREATED",
            (w // 2 - 180, h // 2 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"SAVED {max_samples} SAMPLES TO DATABASE. SECURED.",
            (w // 2 - 190, h // 2 + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1,
            lineType=cv2.LINE_AA
        )

        show_fullscreen("Register Member", frame)
        cv2.waitKey(1)

        if time.time() - success_timer > 2.0:
            break

    show_fullscreen("Register Member", frame)
    if state != STATE_CAPTURING: # capture state waitKey is handled with a delay above
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
print("Process completed.")