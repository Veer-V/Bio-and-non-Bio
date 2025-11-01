
"""
waste_detector_fusion_fixed.py

Integrated script:
 - Robust TFLite local classifier (handles uint8/float models & variable input size)
 - Optional Hugging Face image-classification pipeline (disabled by default)
 - Local keyword DB logic
 - Serial communication with Arduino (handshake + non-blocking reads)
 - Decision fusion and safe sending of single-character commands: 'B','N','C'
 - Commands are sent when triggered by Arduino distance readings.
"""

import os
import time
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import serial
from serial.serialutil import SerialException
import traceback

# Optional HF - disabled by default to avoid automatic downloads
USE_HF = False

if USE_HF:
    try:
        from transformers import pipeline
        import torch
        HF_AVAILABLE = True
    except Exception:
        HF_AVAILABLE = False
else:
    HF_AVAILABLE = False

# ---------------------------
# CONFIG (edit these for your machine)
# ---------------------------
MODEL_PATH = r"C:\Users\Osama\OneDrive\Desktop\IOT\biodegradable_classifier_model.tflite"
ARDUINO_PORT = "COM6"             # change to your port if needed (e.g., "COM3" or "/dev/ttyUSB0")
ARDUINO_BAUD = 9600
CAMERA_INDEX = 1                  # try 0 first; change to 1 if you use a second camera
DISTANCE_THRESHOLD = 20           # cm - trigger classification when object is closer than this
CONFIDENCE_THRESHOLD = 0.65       # minimum confidence to accept classification
CONF_MARGIN = 0.05                # margin to prefer one model over another
LOCAL_DB_STRONG_CONF = 0.98       # if local keyword DB finds match, treat as near-certain
TOP_K_HF = 8                      # how many HF top labels to check for keywords
SHOW_DEBUG = False                # set True to print raw model outputs for debugging
COOLDOWN = 0                     # seconds - minimum time between classifications

# classes (must match your TFLite training order)
CLASS_NAMES = ["Biodegradable", "Non-Biodegradable"]

# ---------------------------
# Local keyword database
# ---------------------------
LOCAL_KEYWORD_DB = {
    # biodegradable
    "apple": 0, "banana": 0, "orange": 0, "vegetable": 0, "fruit": 0,
    "leaf": 0, "leaves": 0, "grass": 0, "wood": 0, "paper": 0,
    "cardboard": 0, "bread": 0, "rice": 0, "pasta": 0, "egg": 0,
    "meat": 0, "fish": 0, "chicken": 0, "cotton": 0, "wool": 0,
    # non-biodegradable
    "plastic": 1, "bottle": 1, "can": 1, "metal": 1, "glass": 1,
    "battery": 1, "electronic": 1, "phone": 1, "computer": 1,
    "tv": 1, "wire": 1, "cable": 1, "styrofoam": 1, "polyester": 1,
    "nylon": 1, "paint": 1, "oil": 1
}

# ---------------------------
# Utility functions
# ---------------------------
def keyword_db_lookup(label_text):
    if not label_text:
        return None, 0.0
    text = label_text.lower()
    for token, cls in LOCAL_KEYWORD_DB.items():
        if token in text:
            return cls, LOCAL_DB_STRONG_CONF
    return None, 0.0

def detect_object_presence(frame_bgr, variance_threshold=100.0):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > variance_threshold

# ---------------------------
# Hugging Face wrapper (optional)
# ---------------------------
def initialize_hf_pipeline():
    if not HF_AVAILABLE:
        print("Hugging Face Transformers not available (transformers/torch missing) or USE_HF=False.")
        return None
    try:
        print("Initializing Hugging Face image-classification pipeline...")
        device = 0 if torch.cuda.is_available() else -1
        hf = pipeline("image-classification", model="google/vit-base-patch16-224", device=device)
        print("Hugging Face pipeline ready.")
        return hf
    except Exception as e:
        print("Could not initialize Hugging Face pipeline:", e)
        traceback.print_exc()
        return None

def classify_with_hf(hf_pipeline, image_np, top_k=TOP_K_HF):
    try:
        pil_img = Image.fromarray(image_np)
        results = hf_pipeline(pil_img, top_k=top_k)
        bio_score = 0.0
        non_score = 0.0
        best_label = None
        best_score = 0.0

        for r in results:
            lbl = r.get("label", "").lower()
            sc = float(r.get("score", 0.0))
            if sc > best_score:
                best_score = sc
                best_label = r.get("label", "")
            for k, v in LOCAL_KEYWORD_DB.items():
                if k in lbl:
                    if v == 0:
                        bio_score += sc
                    else:
                        non_score += sc
        if bio_score == 0 and non_score == 0:
            top_label_lower = (best_label or "").lower()
            if any(tok in top_label_lower for tok in ["plastic", "bottle", "metal", "glass", "battery", "phone", "computer", "can", "styrofoam"]):
                return 1, best_score, best_label, results
            if any(tok in top_label_lower for tok in ["food", "apple", "banana", "vegetable", "fruit", "leaf", "paper", "wood"]):
                return 0, best_score, best_label, results
            return None, best_score, best_label, results
        if bio_score >= non_score:
            return 0, bio_score, best_label, results
        else:
            return 1, non_score, best_label, results
    except Exception as e:
        print("Error during Hugging Face classification:", e)
        traceback.print_exc()
        return None, 0.0, "Error", []

# ---------------------------
# TFLite utilities
# ---------------------------
def initialize_tflite_interpreter(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model file not found at: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_d = interpreter.get_input_details()
    output_d = interpreter.get_output_details()
    return interpreter, input_d, output_d

def prepare_input_for_interpreter(frame_bgr, input_details):
    in_shape = input_details[0]['shape']
    if len(in_shape) == 4:
        _, h, w, c = in_shape
    elif len(in_shape) == 3:
        h, w, c = in_shape
    else:
        h, w, c = 224, 224, 3

    if h <= 0: h = 224
    if w <= 0: w = 224

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    pil = pil.resize((w, h))
    arr = np.array(pil)

    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)  # (1,h,w,3)

    expected_dtype = input_details[0]['dtype']
    if expected_dtype == np.float32:
        arr = arr.astype(np.float32) / 255.0
    elif expected_dtype == np.uint8:
        arr = arr.astype(np.uint8)
    else:
        arr = arr.astype(np.float32) / 255.0

    return arr

def classify_with_tflite(interpreter, input_details, output_details, frame_bgr):
    arr = prepare_input_for_interpreter(frame_bgr, input_details)
    try:
        interpreter.set_tensor(input_details[0]['index'], arr)
    except Exception:
        try:
            interpreter.resize_tensor_input(input_details[0]['index'], arr.shape)
            interpreter.allocate_tensors()
            input_d = interpreter.get_input_details()
            output_d = interpreter.get_output_details()
            interpreter.set_tensor(input_d[0]['index'], arr)
        except Exception as e:
            raise RuntimeError("Failed to set TFLite input tensor: " + str(e))

    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])[0]
    conf = float(np.max(out))
    pred = int(np.argmax(out))
    label = CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else "Unknown"
    return pred, conf, label, out

# ---------------------------
# Serial (Arduino) helper
# ---------------------------
class DummySerial:
    def __init__(self):
        self.in_waiting = 0
    def write(self, data):
        print(f"[DummySerial] write: {data!r}")
    def readline(self):
        return b""
    def close(self):
        print("[DummySerial] close")

def init_arduino(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2)  # allow Arduino to reset
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            ser.flushInput()
            ser.flushOutput()
        print(f"✅ Connected to Arduino on {port} @ {baud} baud")
        return ser
    except SerialException as e:
        print(f"⚠ Could not open serial port {port}: {e}")
        print("Falling back to DummySerial (no Arduino).")
        return DummySerial()
    except Exception as e:
        print("Unexpected error opening serial:", e)
        traceback.print_exc()
        return DummySerial()

def send_to_arduino(ser, command, wait_ack=True, ack_timeout=2.0):
    try:
        if isinstance(ser, DummySerial):
            ser.write(command.encode())
            print("DummySerial: pretending ACK received.")
            return True

        ser.write(command.encode())
        print(f"📤 Sent command to Arduino: {command!r}")

        if not wait_ack:
            return True

        start = time.time()
        while time.time() - start < ack_timeout:
            try:
                line = ser.readline()
            except Exception:
                line = b""
            if not line:
                time.sleep(0.03)
                continue
            try:
                text = line.decode(errors="ignore").strip()
            except Exception:
                text = str(line)
            if text:
                print(f"📥 Arduino: {text}")
                if text.startswith("ACK:"):
                    # ack format expected ACK:X where X is the command char
                    if len(text) >= 5 and text[4] == command:
                        return True
                    # also support ACK:<cmd> format like ACK:C
                    if text.endswith(command):
                        return True
                    else:
                        print("⚠ Received ACK for a different command:", text)
                        return False
        print("⚠ No ACK from Arduino (timeout).")
        return False
    except Exception as e:
        print("⚠ Error communicating with Arduino:", e)
        traceback.print_exc()
        return False

def read_arduino_nonblocking(ser, last_state):
    try:
        if isinstance(ser, DummySerial):
            return last_state
        try:
            available = ser.in_waiting
        except Exception:
            available = 0
        while available:
            line = ser.readline()
            try:
                text = line.decode(errors="ignore").strip()
            except Exception:
                text = str(line)
            if not text:
                try:
                    available = ser.in_waiting
                except Exception:
                    break
                continue
            if SHOW_DEBUG:
                print("[Arduino] >", text)
            if text.upper().startswith("DISTANCE:"):
                try:
                    # split on colon and parse
                    dist_str = text.split(":",1)[1].strip()
                    distance = float(dist_str)
                    last_state['last_distance'] = distance
                    last_state['last_distance_ts'] = time.time()
                except (ValueError, IndexError):
                    print("Invalid distance format:", text)
            elif text == "BUTTON_PRESSED":
                last_state['button_pressed'] = True
                last_state['button_ts'] = time.time()
            elif text.startswith("ACK:"):
                last_state['last_ack'] = text
            else:
                last_state['last_msg'] = text

            try:
                available = ser.in_waiting
            except Exception:
                break
        return last_state
    except Exception as e:
        print("Error reading Arduino serial:", e)
        return last_state

# ---------------------------
# Decision fusion logic
# ---------------------------
def fuse_decisions(local_cls, local_conf, local_label, hf_cls, hf_conf, hf_label):
    if local_cls is None and hf_cls is None:
        return None, 0.0, "Unknown", "no_info"
    if local_cls is not None and hf_cls is not None:
        if local_cls == hf_cls:
            avg_conf = (local_conf + hf_conf) / 2.0
            return local_cls, avg_conf, CLASS_NAMES[local_cls], "agree_avg"
        if local_conf >= hf_conf + CONF_MARGIN:
            return local_cls, local_conf, local_label, "local_higher"
        if hf_conf >= local_conf + CONF_MARGIN:
            return hf_cls, hf_conf, hf_label, "hf_higher"
        if local_conf >= hf_conf:
            return local_cls, local_conf, local_label, "local_higher_close"
        else:
            return hf_cls, hf_conf, hf_label, "hf_higher_close"
    if hf_cls is not None:
        return hf_cls, hf_conf, hf_label, "hf_only"
    return local_cls, local_conf, local_label, "local_only"

# ---------------------------
# Main
# ---------------------------
def main():
    print("Starting Waste Detection Fusion System")
    try:
        interpreter, input_details, output_details = initialize_tflite_interpreter(MODEL_PATH)
        print("TFLite model loaded.")
        if SHOW_DEBUG:
            print("Input details:", input_details)
            print("Output details:", output_details)
    except Exception as e:
        print("Failed to load TFLite model:", e)
        traceback.print_exc()
        return

    hf_pipeline = None
    if HF_AVAILABLE and USE_HF:
        try:
            hf_pipeline = initialize_hf_pipeline()
        except Exception as e:
            hf_pipeline = None

    arduino = init_arduino(ARDUINO_PORT, ARDUINO_BAUD)

    arduino_state = {'last_distance': None, 'last_distance_ts': None, 'last_ack': None, 'last_msg': None}

    last_capture_time = 0

    cap_flag = cv2.CAP_DSHOW if os.name == "nt" else 0
    cap = cv2.VideoCapture(CAMERA_INDEX, cap_flag)
    if not cap.isOpened():
        print("Could not open camera index", CAMERA_INDEX)
        return

    print("Ready. Distance-based triggering active. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        arduino_state = read_arduino_nonblocking(arduino, arduino_state)

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        distance = arduino_state.get('last_distance')
        # ensure distance is valid (non-negative) and less than threshold
        if distance is not None and distance >= 0 and distance < DISTANCE_THRESHOLD and time.time() - last_capture_time > COOLDOWN:
            print(f"🔍 Object detected at {distance:.1f} cm. Triggering classification...")
            last_capture_time = time.time()

            object_present = detect_object_presence(frame)
            if not object_present:
                result_text = "⚠ No object detected in image"
                print(result_text)
                display_frame = frame.copy()
                cv2.putText(display_frame, result_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow("Result", display_frame)
                cv2.waitKey(1500)
                continue

            try:
                tflite_pred, tflite_conf, tflite_label, tflite_raw = classify_with_tflite(
                    interpreter, input_details, output_details, frame
                )
                if SHOW_DEBUG:
                    print("TFLite:", tflite_pred, tflite_conf, tflite_label)

                db_cls1, db_conf1 = keyword_db_lookup(tflite_label)
                if db_cls1 is not None:
                    local_cls = db_cls1
                    local_conf = db_conf1
                    local_label = CLASS_NAMES[local_cls]
                else:
                    local_cls = tflite_pred if (tflite_conf >= 0.0) else None
                    local_conf = tflite_conf
                    local_label = tflite_label

                if hf_pipeline:
                    hf_cls, hf_conf, hf_label, hf_raw = classify_with_hf(hf_pipeline, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if SHOW_DEBUG:
                        print("HF:", hf_cls, hf_conf, hf_label)
                    db_cls2, db_conf2 = keyword_db_lookup(hf_label)
                    if db_cls2 is not None and local_cls is None:
                        local_cls = db_cls2
                        local_conf = db_conf2
                        local_label = CLASS_NAMES[db_cls2]
                else:
                    hf_cls, hf_conf, hf_label, hf_raw = None, 0.0, "Unknown", []

                final_cls, final_conf, final_label, reason = fuse_decisions(
                    local_cls, local_conf, local_label,
                    hf_cls, hf_conf, hf_label
                )

                if (final_cls is None or final_conf < CONFIDENCE_THRESHOLD) and hf_pipeline:
                    final_cls, final_conf, final_label, reason = hf_cls, hf_conf, hf_label, "hf_fallback"

                if final_cls is None or final_conf < CONFIDENCE_THRESHOLD:
                    result_text = "⚠ Unable to classify object"
                    print(result_text)
                    cmd_char = 'C'  # Unknown object -> matches Arduino ACK:C
                    sent_ok = send_to_arduino(arduino, cmd_char)
                    if not sent_ok:
                        print("Warning: Arduino did not ACK command.")
                else:
                    dustbin = 1 if final_cls == 0 else 2
                    result_text = f"{final_label} → Dustbin {dustbin} (Conf: {final_conf:.2f})"
                    print(f"✅ {result_text}    [decision_reason={reason}]")
                    time.sleep(0.3)
                    cmd_char = 'B' if final_cls == 0 else 'N'
                    sent_ok = send_to_arduino(arduino, cmd_char)
                    if not sent_ok:
                        print("Warning: Arduino did not ACK command.")

                display_frame = frame.copy()
                color = (200, 200, 200)
                if final_cls == 0:
                    color = (0, 255, 0)
                elif final_cls == 1:
                    color = (0, 165, 255)
                cv2.putText(display_frame, result_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cmd_sent_str = 'B' if (final_cls == 0) else ('N' if final_cls == 1 else 'C')
                cv2.putText(display_frame, f"Cmd: {cmd_sent_str}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.imshow("Result", display_frame)
                cv2.waitKey(1500)

            except Exception as e:
                print("Error during classification cycle:", e)
                traceback.print_exc()

        elif key == ord("q"):
            print("Quitting...")
            break

        arduino_state = read_arduino_nonblocking(arduino, arduino_state)

    cap.release()
    cv2.destroyAllWindows()
    try:
        arduino.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()

"""
waste_detector_fusion_fixed.py

Integrated script:
 - Robust TFLite local classifier (handles uint8/float models & variable input size)
 - Optional Hugging Face image-classification pipeline (disabled by default)
 - Local keyword DB logic
 - Serial communication with Arduino (handshake + non-blocking reads)
 - Decision fusion and safe sending of single-character commands: 'B','N','C'
 - Commands are sent when triggered by Arduino distance readings.
"""

import os
import time
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import serial
from serial.serialutil import SerialException
import traceback

# Optional HF - disabled by default to avoid automatic downloads
USE_HF = False

if USE_HF:
    try:
        from transformers import pipeline
        import torch
        HF_AVAILABLE = True
    except Exception:
        HF_AVAILABLE = False
else:
    HF_AVAILABLE = False

# ---------------------------
# CONFIG (edit these for your machine)
# ---------------------------
MODEL_PATH = r"C:\Users\Osama\OneDrive\Desktop\IOT\biodegradable_classifier_model.tflite"
ARDUINO_PORT = "COM6"             # change to your port if needed (e.g., "COM3" or "/dev/ttyUSB0")
ARDUINO_BAUD = 9600
CAMERA_INDEX = 1                  # try 0 first; change to 1 if you use a second camera
DISTANCE_THRESHOLD = 20           # cm - trigger classification when object is closer than this
CONFIDENCE_THRESHOLD = 0.65       # minimum confidence to accept classification
CONF_MARGIN = 0.05                # margin to prefer one model over another
LOCAL_DB_STRONG_CONF = 0.98       # if local keyword DB finds match, treat as near-certain
TOP_K_HF = 8                      # how many HF top labels to check for keywords
SHOW_DEBUG = False                # set True to print raw model outputs for debugging
COOLDOWN = 0                     # seconds - minimum time between classifications

# classes (must match your TFLite training order)
CLASS_NAMES = ["Biodegradable", "Non-Biodegradable"]

# ---------------------------
# Local keyword database
# ---------------------------
LOCAL_KEYWORD_DB = {
    # biodegradable
    "apple": 0, "banana": 0, "orange": 0, "vegetable": 0, "fruit": 0,
    "leaf": 0, "leaves": 0, "grass": 0, "wood": 0, "paper": 0,
    "cardboard": 0, "bread": 0, "rice": 0, "pasta": 0, "egg": 0,
    "meat": 0, "fish": 0, "chicken": 0, "cotton": 0, "wool": 0,
    # non-biodegradable
    "plastic": 1, "bottle": 1, "can": 1, "metal": 1, "glass": 1,
    "battery": 1, "electronic": 1, "phone": 1, "computer": 1,
    "tv": 1, "wire": 1, "cable": 1, "styrofoam": 1, "polyester": 1,
    "nylon": 1, "paint": 1, "oil": 1
}

# ---------------------------
# Utility functions
# ---------------------------
def keyword_db_lookup(label_text):
    if not label_text:
        return None, 0.0
    text = label_text.lower()
    for token, cls in LOCAL_KEYWORD_DB.items():
        if token in text:
            return cls, LOCAL_DB_STRONG_CONF
    return None, 0.0

def detect_object_presence(frame_bgr, variance_threshold=100.0):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > variance_threshold

# ---------------------------
# Hugging Face wrapper (optional)
# ---------------------------
def initialize_hf_pipeline():
    if not HF_AVAILABLE:
        print("Hugging Face Transformers not available (transformers/torch missing) or USE_HF=False.")
        return None
    try:
        print("Initializing Hugging Face image-classification pipeline...")
        device = 0 if torch.cuda.is_available() else -1
        hf = pipeline("image-classification", model="google/vit-base-patch16-224", device=device)
        print("Hugging Face pipeline ready.")
        return hf
    except Exception as e:
        print("Could not initialize Hugging Face pipeline:", e)
        traceback.print_exc()
        return None

def classify_with_hf(hf_pipeline, image_np, top_k=TOP_K_HF):
    try:
        pil_img = Image.fromarray(image_np)
        results = hf_pipeline(pil_img, top_k=top_k)
        bio_score = 0.0
        non_score = 0.0
        best_label = None
        best_score = 0.0

        for r in results:
            lbl = r.get("label", "").lower()
            sc = float(r.get("score", 0.0))
            if sc > best_score:
                best_score = sc
                best_label = r.get("label", "")
            for k, v in LOCAL_KEYWORD_DB.items():
                if k in lbl:
                    if v == 0:
                        bio_score += sc
                    else:
                        non_score += sc
        if bio_score == 0 and non_score == 0:
            top_label_lower = (best_label or "").lower()
            if any(tok in top_label_lower for tok in ["plastic", "bottle", "metal", "glass", "battery", "phone", "computer", "can", "styrofoam"]):
                return 1, best_score, best_label, results
            if any(tok in top_label_lower for tok in ["food", "apple", "banana", "vegetable", "fruit", "leaf", "paper", "wood"]):
                return 0, best_score, best_label, results
            return None, best_score, best_label, results
        if bio_score >= non_score:
            return 0, bio_score, best_label, results
        else:
            return 1, non_score, best_label, results
    except Exception as e:
        print("Error during Hugging Face classification:", e)
        traceback.print_exc()
        return None, 0.0, "Error", []

# ---------------------------
# TFLite utilities
# ---------------------------
def initialize_tflite_interpreter(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model file not found at: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_d = interpreter.get_input_details()
    output_d = interpreter.get_output_details()
    return interpreter, input_d, output_d

def prepare_input_for_interpreter(frame_bgr, input_details):
    in_shape = input_details[0]['shape']
    if len(in_shape) == 4:
        _, h, w, c = in_shape
    elif len(in_shape) == 3:
        h, w, c = in_shape
    else:
        h, w, c = 224, 224, 3

    if h <= 0: h = 224
    if w <= 0: w = 224

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    pil = pil.resize((w, h))
    arr = np.array(pil)

    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)  # (1,h,w,3)

    expected_dtype = input_details[0]['dtype']
    if expected_dtype == np.float32:
        arr = arr.astype(np.float32) / 255.0
    elif expected_dtype == np.uint8:
        arr = arr.astype(np.uint8)
    else:
        arr = arr.astype(np.float32) / 255.0

    return arr

def classify_with_tflite(interpreter, input_details, output_details, frame_bgr):
    arr = prepare_input_for_interpreter(frame_bgr, input_details)
    try:
        interpreter.set_tensor(input_details[0]['index'], arr)
    except Exception:
        try:
            interpreter.resize_tensor_input(input_details[0]['index'], arr.shape)
            interpreter.allocate_tensors()
            input_d = interpreter.get_input_details()
            output_d = interpreter.get_output_details()
            interpreter.set_tensor(input_d[0]['index'], arr)
        except Exception as e:
            raise RuntimeError("Failed to set TFLite input tensor: " + str(e))

    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])[0]
    conf = float(np.max(out))
    pred = int(np.argmax(out))
    label = CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else "Unknown"
    return pred, conf, label, out

# ---------------------------
# Serial (Arduino) helper
# ---------------------------
class DummySerial:
    def __init__(self):
        self.in_waiting = 0
    def write(self, data):
        print(f"[DummySerial] write: {data!r}")
    def readline(self):
        return b""
    def close(self):
        print("[DummySerial] close")

def init_arduino(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2)  # allow Arduino to reset
        try:
            ser.reset_input_buffer()
            ser.reset_output_buffer()
        except Exception:
            ser.flushInput()
            ser.flushOutput()
        print(f"✅ Connected to Arduino on {port} @ {baud} baud")
        return ser
    except SerialException as e:
        print(f"⚠ Could not open serial port {port}: {e}")
        print("Falling back to DummySerial (no Arduino).")
        return DummySerial()
    except Exception as e:
        print("Unexpected error opening serial:", e)
        traceback.print_exc()
        return DummySerial()

def send_to_arduino(ser, command, wait_ack=True, ack_timeout=2.0):
    try:
        if isinstance(ser, DummySerial):
            ser.write(command.encode())
            print("DummySerial: pretending ACK received.")
            return True

        ser.write(command.encode())
        print(f"📤 Sent command to Arduino: {command!r}")

        if not wait_ack:
            return True

        start = time.time()
        while time.time() - start < ack_timeout:
            try:
                line = ser.readline()
            except Exception:
                line = b""
            if not line:
                time.sleep(0.03)
                continue
            try:
                text = line.decode(errors="ignore").strip()
            except Exception:
                text = str(line)
            if text:
                print(f"📥 Arduino: {text}")
                if text.startswith("ACK:"):
                    # ack format expected ACK:X where X is the command char
                    if len(text) >= 5 and text[4] == command:
                        return True
                    # also support ACK:<cmd> format like ACK:C
                    if text.endswith(command):
                        return True
                    else:
                        print("⚠ Received ACK for a different command:", text)
                        return False
        print("⚠ No ACK from Arduino (timeout).")
        return False
    except Exception as e:
        print("⚠ Error communicating with Arduino:", e)
        traceback.print_exc()
        return False

def read_arduino_nonblocking(ser, last_state):
    try:
        if isinstance(ser, DummySerial):
            return last_state
        try:
            available = ser.in_waiting
        except Exception:
            available = 0
        while available:
            line = ser.readline()
            try:
                text = line.decode(errors="ignore").strip()
            except Exception:
                text = str(line)
            if not text:
                try:
                    available = ser.in_waiting
                except Exception:
                    break
                continue
            if SHOW_DEBUG:
                print("[Arduino] >", text)
            if text.upper().startswith("DISTANCE:"):
                try:
                    # split on colon and parse
                    dist_str = text.split(":",1)[1].strip()
                    distance = float(dist_str)
                    last_state['last_distance'] = distance
                    last_state['last_distance_ts'] = time.time()
                except (ValueError, IndexError):
                    print("Invalid distance format:", text)
            elif text == "BUTTON_PRESSED":
                last_state['button_pressed'] = True
                last_state['button_ts'] = time.time()
            elif text.startswith("ACK:"):
                last_state['last_ack'] = text
            else:
                last_state['last_msg'] = text

            try:
                available = ser.in_waiting
            except Exception:
                break
        return last_state
    except Exception as e:
        print("Error reading Arduino serial:", e)
        return last_state

# ---------------------------
# Decision fusion logic
# ---------------------------
def fuse_decisions(local_cls, local_conf, local_label, hf_cls, hf_conf, hf_label):
    if local_cls is None and hf_cls is None:
        return None, 0.0, "Unknown", "no_info"
    if local_cls is not None and hf_cls is not None:
        if local_cls == hf_cls:
            avg_conf = (local_conf + hf_conf) / 2.0
            return local_cls, avg_conf, CLASS_NAMES[local_cls], "agree_avg"
        if local_conf >= hf_conf + CONF_MARGIN:
            return local_cls, local_conf, local_label, "local_higher"
        if hf_conf >= local_conf + CONF_MARGIN:
            return hf_cls, hf_conf, hf_label, "hf_higher"
        if local_conf >= hf_conf:
            return local_cls, local_conf, local_label, "local_higher_close"
        else:
            return hf_cls, hf_conf, hf_label, "hf_higher_close"
    if hf_cls is not None:
        return hf_cls, hf_conf, hf_label, "hf_only"
    return local_cls, local_conf, local_label, "local_only"

# ---------------------------
# Main
# ---------------------------
def main():
    print("Starting Waste Detection Fusion System")
    try:
        interpreter, input_details, output_details = initialize_tflite_interpreter(MODEL_PATH)
        print("TFLite model loaded.")
        if SHOW_DEBUG:
            print("Input details:", input_details)
            print("Output details:", output_details)
    except Exception as e:
        print("Failed to load TFLite model:", e)
        traceback.print_exc()
        return

    hf_pipeline = None
    if HF_AVAILABLE and USE_HF:
        try:
            hf_pipeline = initialize_hf_pipeline()
        except Exception as e:
            hf_pipeline = None

    arduino = init_arduino(ARDUINO_PORT, ARDUINO_BAUD)

    arduino_state = {'last_distance': None, 'last_distance_ts': None, 'last_ack': None, 'last_msg': None}

    last_capture_time = 0

    cap_flag = cv2.CAP_DSHOW if os.name == "nt" else 0
    cap = cv2.VideoCapture(CAMERA_INDEX, cap_flag)
    if not cap.isOpened():
        print("Could not open camera index", CAMERA_INDEX)
        return

    print("Ready. Distance-based triggering active. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        arduino_state = read_arduino_nonblocking(arduino, arduino_state)

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        distance = arduino_state.get('last_distance')
        # ensure distance is valid (non-negative) and less than threshold
        if distance is not None and distance >= 0 and distance < DISTANCE_THRESHOLD and time.time() - last_capture_time > COOLDOWN:
            print(f"🔍 Object detected at {distance:.1f} cm. Triggering classification...")
            last_capture_time = time.time()

            object_present = detect_object_presence(frame)
            if not object_present:
                result_text = "⚠ No object detected in image"
                print(result_text)
                display_frame = frame.copy()
                cv2.putText(display_frame, result_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow("Result", display_frame)
                cv2.waitKey(1500)
                continue

            try:
                tflite_pred, tflite_conf, tflite_label, tflite_raw = classify_with_tflite(
                    interpreter, input_details, output_details, frame
                )
                if SHOW_DEBUG:
                    print("TFLite:", tflite_pred, tflite_conf, tflite_label)

                db_cls1, db_conf1 = keyword_db_lookup(tflite_label)
                if db_cls1 is not None:
                    local_cls = db_cls1
                    local_conf = db_conf1
                    local_label = CLASS_NAMES[local_cls]
                else:
                    local_cls = tflite_pred if (tflite_conf >= 0.0) else None
                    local_conf = tflite_conf
                    local_label = tflite_label

                if hf_pipeline:
                    hf_cls, hf_conf, hf_label, hf_raw = classify_with_hf(hf_pipeline, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if SHOW_DEBUG:
                        print("HF:", hf_cls, hf_conf, hf_label)
                    db_cls2, db_conf2 = keyword_db_lookup(hf_label)
                    if db_cls2 is not None and local_cls is None:
                        local_cls = db_cls2
                        local_conf = db_conf2
                        local_label = CLASS_NAMES[db_cls2]
                else:
                    hf_cls, hf_conf, hf_label, hf_raw = None, 0.0, "Unknown", []

                final_cls, final_conf, final_label, reason = fuse_decisions(
                    local_cls, local_conf, local_label,
                    hf_cls, hf_conf, hf_label
                )

                if (final_cls is None or final_conf < CONFIDENCE_THRESHOLD) and hf_pipeline:
                    final_cls, final_conf, final_label, reason = hf_cls, hf_conf, hf_label, "hf_fallback"

                if final_cls is None or final_conf < CONFIDENCE_THRESHOLD:
                    result_text = "⚠ Unable to classify object"
                    print(result_text)
                    cmd_char = 'C'  # Unknown object -> matches Arduino ACK:C
                    sent_ok = send_to_arduino(arduino, cmd_char)
                    if not sent_ok:
                        print("Warning: Arduino did not ACK command.")
                else:
                    dustbin = 1 if final_cls == 0 else 2
                    result_text = f"{final_label} → Dustbin {dustbin} (Conf: {final_conf:.2f})"
                    print(f"✅ {result_text}    [decision_reason={reason}]")
                    time.sleep(0.3)
                    cmd_char = 'B' if final_cls == 0 else 'N'
                    sent_ok = send_to_arduino(arduino, cmd_char)
                    if not sent_ok:
                        print("Warning: Arduino did not ACK command.")

                display_frame = frame.copy()
                color = (200, 200, 200)
                if final_cls == 0:
                    color = (0, 255, 0)
                elif final_cls == 1:
                    color = (0, 165, 255)
                cv2.putText(display_frame, result_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cmd_sent_str = 'B' if (final_cls == 0) else ('N' if final_cls == 1 else 'C')
                cv2.putText(display_frame, f"Cmd: {cmd_sent_str}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.imshow("Result", display_frame)
                cv2.waitKey(1500)

            except Exception as e:
                print("Error during classification cycle:", e)
                traceback.print_exc()

        elif key == ord("q"):
            print("Quitting...")
            break

        arduino_state = read_arduino_nonblocking(arduino, arduino_state)

    cap.release()
    cv2.destroyAllWindows()
    try:
        arduino.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()

