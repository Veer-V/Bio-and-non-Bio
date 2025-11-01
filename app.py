from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
from datetime import datetime, timedelta
import json
import time
import threading
import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
import io
import serial
import requests
import atexit

# Import our custom modules
# NOTE: You MUST ensure 'models.py' and 'analytics.py' are available and correct in your environment.
# They are assumed to be correct for this Flask file to run.
from models import db, WasteCollection, ContainerStatus, SystemStats
# Corrected import: Removed set_arduino_port which is no longer defined in iot_handler.py
from iot_handler import initialize_arduino, cleanup_arduino, arduino_handler
from analytics import waste_analytics

# ---------------------------
# Local keyword database (from waste_detector_fusion_fixed.py)
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
# Configuration constants (from waste_detector_fusion_fixed.py)
# ---------------------------
DISTANCE_THRESHOLD = 20           # cm - trigger classification when object is closer than this
CONFIDENCE_THRESHOLD = 0.65       # minimum confidence to accept classification
CONF_MARGIN = 0.05                # margin to prefer one model over another
LOCAL_DB_STRONG_CONF = 0.98       # if local keyword DB finds match, treat as near-certain
TOP_K_HF = 8                      # how many HF top labels to check for keywords
CLASS_NAMES = ["Biodegradable", "Non-Biodegradable"]

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///waste_management.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for IoT integration
# NOTE: arduino_handler is imported globally from iot_handler.py and is the central connection manager.
model_interpreter = None
camera_active = False
system_start_time = datetime.now()

# Initialize database (must be done within the Flask application context)
with app.app_context():
    db.create_all()

# Utility Functions

def initialize_tflite_model():
    """Initialize TFLite model."""
    try:
        # NOTE: Update this path to your actual TFLite model location
        model_path = r"C:\Users\Osama\OneDrive\Desktop\AI And IOT\biodegradable_classifier_model.tflite"
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        return None

def load_and_preprocess_image(img_data, img_size=(224, 224)):
    """Convert image data to TFLite input format."""
    # Convert base64 to numpy array
    img_bytes = base64.b64decode(img_data.split(',')[1])
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img = pil_img.resize(img_size)
    img_array = np.array(pil_img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_with_huggingface_api(image_data):
    """Classify image using Hugging Face API with keyword-based decision fusion."""
    try:
        # Hugging Face API endpoint for image classification
        api_url = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224-in21k"
        headers = {
            # NOTE: Replace with your actual Hugging Face API Token
            "Authorization": f"Bearer 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",
            "Content-Type": "application/json"
        }

        # Convert base64 to image bytes
        image_bytes = base64.b64decode(image_data.split(',')[1])

        # Prepare the payload
        # Note: Sending binary data as a POST request body
        response = requests.post(api_url, headers=headers, data=image_bytes, timeout=30)

        if response.status_code == 200:
            result = response.json()

            # Process the classification results with keyword fusion
            if result and isinstance(result, list) and len(result) > 0:
                predictions = result # Assuming the response is a list of predictions

                # Check local keyword database first
                local_match = None
                local_conf = 0.0

                for pred in predictions[:TOP_K_HF]:
                    label = pred.get('label', '').lower()
                    # Clean up common model label artifacts (like "n01234567")
                    if ',' in label:
                        label = label.split(',')[0].strip()
                    score = pred.get('score', 0.0)

                    for keyword, class_id in LOCAL_KEYWORD_DB.items():
                        if keyword in label or label == keyword:
                            local_match = class_id
                            local_conf = LOCAL_DB_STRONG_CONF  # Strong confidence for keyword match
                            print(f"🔍 Local keyword match: '{keyword}' in '{label}' -> {CLASS_NAMES[class_id]}")
                            break

                    if local_match is not None:
                        break

                # If local keyword found, use it
                if local_match is not None:
                    return local_match, local_conf, CLASS_NAMES[local_match]

                # Otherwise, use traditional keyword scoring
                bio_score = 0.0
                non_bio_score = 0.0

                biodegradable_keywords = [
                    'food', 'vegetable', 'fruit', 'plant', 'organic', 'paper', 'wood',
                    'leaf', 'natural', 'biodegradable', 'compost', 'recyclable', 'bread', 'rice', 'meat', 'egg'
                ]

                non_biodegradable_keywords = [
                    'plastic', 'metal', 'glass', 'synthetic', 'chemical', 'electronic',
                    'battery', 'toxic', 'hazardous', 'non-biodegradable', 'can', 'bottle', 'styrofoam', 'nylon'
                ]

                # Tally scores for top 10 predictions
                for pred in predictions[:10]:
                    label = pred.get('label', '').lower()
                    score = pred.get('score', 0.0)

                    for keyword in biodegradable_keywords:
                        if keyword in label:
                            bio_score += score
                            break

                    for keyword in non_biodegradable_keywords:
                        if keyword in label:
                            non_bio_score += score
                            break
                
                # Decision based on accumulated score
                if bio_score > non_bio_score:
                    return 0, float(bio_score), "Biodegradable"
                elif non_bio_score > bio_score:
                    return 1, float(non_bio_score), "Non-Biodegradable"
                else:
                    # Default to biodegradable if scores are equal (neutral item)
                    return 0, 0.0, "Biodegradable (Neutral Score)"

            # Fallback if result format is unexpected
            return None, 0.0, "Error: Unexpected HF response format"
        
        else:
            print(f"Hugging Face API error: {response.status_code} - {response.text}")
            return None, 0.0, "Error"

    except Exception as e:
        print(f"Error with Hugging Face API: {e}")
        return None, 0.0, "Error"

def classify_with_local_model(interpreter, image_data):
    """Classify image using local TFLite model."""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        img_array = load_and_preprocess_image(image_data)
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        confidence = float(np.max(prediction))  # Convert numpy float32 to Python float
        predicted_class = int(np.argmax(prediction))  # Convert numpy int to Python int

        class_names = ['Biodegradable', 'Non-Biodegradable']
        return predicted_class, confidence, class_names[predicted_class]

    except Exception as e:
        print(f"Error with local model: {e}")
        return None, 0.0, "Error"

def send_to_arduino(command):
    """Send command to Arduino, delegating connection logic to the handler."""
    global arduino_handler

    # Check if handler is connected; if not, handler attempts reconnection internally on send.
    
    # Ensure proper command format
    if not command.endswith('\n'):
        command += '\n'

    # The ArduinoHandler.send_command encapsulates connection and retry logic.
    success = arduino_handler.send_command(command)
    
    # If initial send or internal re-send failed, check the handler's final state
    if not success and not arduino_handler.connected:
        print("❌ Failed to send command after handler-level reconnection attempts.")
        return False
        
    return success

def check_huggingface_api():
    """Check if Hugging Face API is available."""
    try:
        # Simple test request to check API availability
        api_url = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224-in21k"
        headers = {
            "Authorization": f"Bearer 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"
        }

        # Make a simple GET request to check if API is accessible
        response = requests.get(api_url, headers=headers, timeout=10)

        # API is available if we get a 200 (healthy) or 503 (model loading/warming up)
        return response.status_code in [200, 503]

    except Exception as e:
        # DNS error, network issue, or timeout
        print(f"Error checking Hugging Face API: {e}")
        return False

def update_statistics(item_type, confidence, method):
    """Update system statistics."""
    try:
        today = datetime.now().date()

        # Update or create daily stats
        stats = SystemStats.query.filter_by(date=today).first()
        if not stats:
            stats = SystemStats(date=today)
            db.session.add(stats)

        stats.total_collections += 1
        if item_type == 'Biodegradable':
            stats.biodegradable_count += 1
        else:
            stats.non_biodegradable_count += 1

        # Update uptime
        uptime = (datetime.now() - system_start_time).total_seconds() / 3600
        stats.uptime_hours = uptime

        db.session.commit()

        # Emit real-time update
        socketio.emit('stats_update', {
            'total_collections': stats.total_collections,
            'biodegradable_count': stats.biodegradable_count,
            'non_biodegradable_count': stats.non_biodegradable_count,
            'uptime_hours': round(uptime, 2)
        })

    except Exception as e:
        print(f"Error updating statistics: {e}")

# Routes
@app.route('/')
def index():
    """Main dashboard."""
    # NOTE: You need to create 'index.html', 'camera.html', 'statistics.html', etc.
    return render_template('index.html')

@app.route('/camera')
def camera():
    """Camera interface."""
    return render_template('camera.html')

@app.route('/statistics')
def statistics():
    """Statistics page."""
    return render_template('statistics.html')

@app.route('/mobile')
def mobile():
    """Mobile interface."""
    return render_template('mobile.html')

@app.route('/tablet')
def tablet():
    """Tablet interface."""
    return render_template('tablet.html')

@app.route('/desktop')
def desktop():
    """Desktop interface."""
    return render_template('desktop.html')

@app.route('/api/welcome')
def welcome():
    """Welcome endpoint that logs requests and returns a welcome message."""
    print(f"Request received: {request.method} {request.path}")
    return jsonify({"message": "Welcome to the IoT Waste Management System"})

@app.route('/api/classify', methods=['POST'])
def classify_image():
    """API endpoint for image classification with decision fusion."""
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No image provided'}), 400

        # Initialize results
        predicted_class, confidence, label = None, 0.0, "Unknown"
        method = 'fusion'

        # Try LOCAL MODEL FIRST (as preferred method)
        local_result = None
        if model_interpreter:
            print("🤖 Trying local TFLite model first...")
            local_class, local_conf, local_label = classify_with_local_model(model_interpreter, image_data)

            if local_class is not None:
                local_result = (local_class, local_conf, local_label)
                print(f"✅ Local model result: {local_label} (Confidence: {local_conf:.2f})")

        # Try Hugging Face API (with keyword fusion)
        hf_result = None
        print("🤗 Trying Hugging Face API with keyword fusion...")
        hf_class, hf_conf, hf_label = classify_with_huggingface_api(image_data)

        if hf_class is not None:
            hf_result = (hf_class, hf_conf, hf_label)
            print(f"✅ HF result: {hf_label} (Confidence: {hf_conf:.2f})")

        # Decision Fusion Logic
        if local_result and hf_result:
            local_class, local_conf, local_label = local_result
            hf_class, hf_conf, hf_label = hf_result

            # If both agree, use the higher confidence
            if local_class == hf_class:
                if local_conf >= hf_conf:
                    predicted_class, confidence, label = local_class, local_conf, local_label
                    method = 'local_model'
                else:
                    predicted_class, confidence, label = hf_class, hf_conf, hf_label
                    method = 'huggingface_api'
                print(f"🔄 Models agree: Using {method} result")

            # If they disagree, use confidence margin logic
            else:
                conf_diff = abs(local_conf - hf_conf)
                if conf_diff > CONF_MARGIN:
                    # Use the model with higher confidence
                    if local_conf > hf_conf:
                        predicted_class, confidence, label = local_class, local_conf, local_label
                        method = 'local_model'
                    else:
                        predicted_class, confidence, label = hf_class, hf_conf, hf_label
                        method = 'huggingface_api'
                    print(f"⚖️ Models disagree: Using higher confidence ({method})")
                else:
                    # Close confidence, prefer local model
                    predicted_class, confidence, label = local_class, local_conf, local_label
                    method = 'local_model'
                    print("⚖️ Close confidence: Preferring local model")

        elif local_result:
            predicted_class, confidence, label = local_result
            method = 'local_model'
            print("📋 Only local model available")

        elif hf_result:
            predicted_class, confidence, label = hf_result
            method = 'huggingface_api'
            print("📋 Only HF API available")

        else:
            print("❌ Both models failed")
            return jsonify({'error': 'Both classification methods failed'}), 500

        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"⚠️ Low confidence ({confidence:.2f}), but proceeding with classification")

        # Determine container
        container_id = 1 if predicted_class == 0 else 2  # 1 for biodegradable, 2 for non-biodegradable

        # Send command to Arduino
        command = 'B' if predicted_class == 0 else 'N'
        arduino_success = send_to_arduino(command)

        if not arduino_success:
            print("⚠️ Warning: Failed to send command to Arduino")

        # Save to database
        collection = WasteCollection(
            item_type=label,
            confidence=confidence,
            container_id=container_id,
            method=method
        )
        db.session.add(collection)
        db.session.commit()

        # Update container fill level (increment by small amount when item is detected)
        try:
            container = ContainerStatus.query.filter_by(container_id=container_id).first()
            if not container:
                # Assuming default values for new container
                container = ContainerStatus(container_id=container_id, fill_level=0.0, status='empty') 
                db.session.add(container)

            # Increment fill level by 3% for each item (simulate accumulation)
            fill_increment = 3.0
            container.fill_level = min(100.0, container.fill_level + fill_increment)

            # Update status based on new fill level
            if container.fill_level >= 95:
                container.status = 'full'
            elif container.fill_level >= 70:
                container.status = 'filling'
            elif container.fill_level >= 30:
                container.status = 'moderate'
            else:
                container.status = 'empty'

            container.last_updated = datetime.utcnow()
            db.session.commit()

            print(f"📦 Container {container_id} fill level increased to {container.fill_level:.1f}%")

        except Exception as e:
            print(f"❌ Error updating container fill level: {e}")

        # Update statistics
        update_statistics(label, confidence, method)

        # Emit real-time update
        socketio.emit('new_collection', {
            'item_type': label,
            'confidence': round(confidence, 2),
            'container_id': container_id,
            'method': method,
            'timestamp': collection.timestamp.isoformat(),
            'arduino_connected': arduino_handler.connected # Use the handler's status
        })

        return jsonify({
            'prediction': label,
            'confidence': round(confidence, 2),
            'container_id': container_id,
            'method': method,
            'arduino_connected': arduino_handler.connected
        })

    except Exception as e:
        print(f"Error in classification: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    """Get system statistics."""
    try:
        # Get today's stats
        today = datetime.now().date()
        today_stats = SystemStats.query.filter_by(date=today).first()

        # Get recent collections
        recent_collections = WasteCollection.query.order_by(
            WasteCollection.timestamp.desc()
        ).limit(10).all()

        # Get container status
        containers = ContainerStatus.query.all()

        # Get daily collections for the last 7 days
        daily_collections = []
        for i in range(7):
            date = today - timedelta(days=i)
            day_stats = SystemStats.query.filter_by(date=date).first()
            daily_collections.append({
                'date': date.isoformat(),
                'collections': day_stats.total_collections if day_stats else 0
            })

        # Reverse to show oldest to newest
        daily_collections.reverse()

        return jsonify({
            'today_stats': {
                'total_collections': today_stats.total_collections if today_stats else 0,
                'biodegradable_count': today_stats.biodegradable_count if today_stats else 0,
                'non_biodegradable_count': today_stats.non_biodegradable_count if today_stats else 0,
                'uptime_hours': round(today_stats.uptime_hours, 2) if today_stats else 0
            },
            'recent_collections': [{
                'item_type': c.item_type,
                'confidence': c.confidence,
                'container_id': c.container_id,
                'method': c.method,
                'timestamp': c.timestamp.isoformat()
            } for c in recent_collections],
            'containers': [{
                'container_id': c.container_id,
                'fill_level': c.fill_level,
                'status': c.status,
                'last_updated': c.last_updated.isoformat()
            } for c in containers],
            'daily_collections': daily_collections
        })

    except Exception as e:
        print(f"Error getting statistics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system_status')
def get_system_status():
    """Get system status."""
    uptime = (datetime.now() - system_start_time).total_seconds()

    # Check Hugging Face API availability
    huggingface_available = check_huggingface_api()

    return jsonify({
        'uptime_seconds': uptime,
        'uptime_formatted': str(timedelta(seconds=int(uptime))),
        'camera_active': camera_active,
        # Use the handler's connection status
        'arduino_connected': arduino_handler.connected,
        'huggingface_api_available': huggingface_available,
        'local_model_available': model_interpreter is not None
    })

@app.route('/api/status')
def get_status():
    """Get lightweight status for frequent polling."""
    uptime = (datetime.now() - system_start_time).total_seconds()

    return jsonify({
        'uptime_seconds': uptime,
        'camera_active': camera_active,
        # Use the handler's connection status
        'arduino_connected': arduino_handler.connected
    })

@app.route('/api/data')
def get_data():
    """Get dashboard data for real-time updates."""
    try:
        # Get today's stats
        today = datetime.now().date()
        today_stats = SystemStats.query.filter_by(date=today).first()

        # Get recent collections (last 10)
        recent_collections = WasteCollection.query.order_by(
            WasteCollection.timestamp.desc()
        ).limit(10).all()

        # Get container status
        containers = ContainerStatus.query.all()

        return jsonify({
            'stats': {
                'total_collections': today_stats.total_collections if today_stats else 0,
                'biodegradable_count': today_stats.biodegradable_count if today_stats else 0,
                'non_biodegradable_count': today_stats.non_biodegradable_count if today_stats else 0,
                'uptime_hours': round(today_stats.uptime_hours, 2) if today_stats else 0
            },
            'recent_collections': [{
                'item_type': c.item_type,
                'confidence': c.confidence,
                'container_id': c.container_id,
                'method': c.method,
                'timestamp': c.timestamp.isoformat()
            } for c in recent_collections],
            'containers': [{
                'container_id': c.container_id,
                'fill_level': c.fill_level,
                'status': c.status,
                'last_updated': c.last_updated.isoformat()
            } for c in containers]
        })

    except Exception as e:
        print(f"Error getting data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mobile/dashboard')
def get_mobile_dashboard():
    """Get mobile dashboard data."""
    try:
        # Get today's stats
        today = datetime.now().date()
        today_stats = SystemStats.query.filter_by(date=today).first()

        # Get recent collections
        recent_collections = WasteCollection.query.order_by(
            WasteCollection.timestamp.desc()
        ).limit(5).all()

        # Get container status
        containers = ContainerStatus.query.all()

        # Get system status
        uptime = (datetime.now() - system_start_time).total_seconds()

        return jsonify({
            'today_stats': {
                'total_collections': today_stats.total_collections if today_stats else 0,
                'biodegradable_count': today_stats.biodegradable_count if today_stats else 0,
                'non_biodegradable_count': today_stats.non_biodegradable_count if today_stats else 0,
                'uptime_hours': round(today_stats.uptime_hours, 2) if today_stats else 0
            },
            'recent_collections': [{
                'item_type': c.item_type,
                'confidence': c.confidence,
                'container_id': c.container_id,
                'method': c.method,
                'timestamp': c.timestamp.isoformat()
            } for c in recent_collections],
            'containers': [{
                'container_id': c.container_id,
                'fill_level': c.fill_level,
                'status': c.status,
                'last_updated': c.last_updated.isoformat()
            } for c in containers],
            'system_status': {
                'uptime_seconds': uptime,
                'uptime_formatted': str(timedelta(seconds=int(uptime))),
                'camera_active': camera_active,
                'arduino_connected': arduino_handler.connected, # Use the handler's status
                'huggingface_api_available': check_huggingface_api(),
                'local_model_available': model_interpreter is not None
            }
        })

    except Exception as e:
        print(f"Error getting mobile dashboard data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mobile/containers')
def get_mobile_containers():
    """Get mobile container status."""
    try:
        containers = ContainerStatus.query.all()

        return jsonify({
            'containers': [{
                'container_id': c.container_id,
                'fill_level': c.fill_level,
                'status': c.status,
                'last_updated': c.last_updated.isoformat(),
                # Assuming these fields exist in ContainerStatus model
                'capacity': c.capacity,
                'current_weight': c.current_weight
            } for c in containers]
        })

    except Exception as e:
        print(f"Error getting mobile containers: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mobile/open_container', methods=['POST'])
def open_mobile_container():
    """Mobile API to open container."""
    try:
        data = request.get_json()
        container_id_str = data.get('container_id')

        if not container_id_str:
            return jsonify({'error': 'Container ID required'}), 400

        # Convert to integer for consistency
        try:
            container_id = int(container_id_str)
        except ValueError:
            return jsonify({'error': 'Invalid container ID'}), 400

        # Determine the Arduino command based on the container ID
        # Container 1 is Biodegradable (Bio), Container 2 is Non-Biodegradable (Non-Bio)
        # We assume the Arduino sketch expects 'OPEN:1' for Bio and 'OPEN:2' for Non-Bio
        if container_id == 1:
            command = 'OPEN:1'  # Command to open Biodegradable bin
        elif container_id == 2:
            command = 'OPEN:2'  # Command to open Non-Biodegradable bin
        else:
             return jsonify({'error': 'Invalid container ID specified'}), 400

        success = send_to_arduino(command)

        if success:
            # Update container status in database (optional: mark as 'opening' or just update timestamp)
            with app.app_context():
                container = ContainerStatus.query.filter_by(container_id=container_id).first()
                if container:
                    container.last_updated = datetime.utcnow()
                    # Optional: container.status = 'opening'
                    db.session.commit()

            # Emit real-time update
            socketio.emit('container_opened', {
                'container_id': container_id,
                'timestamp': datetime.utcnow().isoformat()
            })

            return jsonify({'success': True, 'message': f'Container {container_id} opened'})
        else:
            return jsonify({'error': 'Failed to open container. Arduino not responsive.'}), 500

    except Exception as e:
        print(f"Error opening container: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mobile/notifications')
def get_mobile_notifications():
    """Get mobile notifications."""
    try:
        # Get containers that need attention
        containers = ContainerStatus.query.filter(
            ContainerStatus.status.in_(['full', 'filling'])
        ).all()

        notifications = []

        for container in containers:
            if container.status == 'full':
                notifications.append({
                    'type': 'warning',
                    'title': 'Container Full',
                    'message': f'Container {container.container_id} is full and needs emptying',
                    'container_id': container.container_id,
                    'timestamp': container.last_updated.isoformat()
                })
            elif container.status == 'filling':
                notifications.append({
                    'type': 'info',
                    'title': 'Container Filling',
                    'message': f'Container {container.container_id} is {container.fill_level:.1f}% full',
                    'container_id': container.container_id,
                    'timestamp': container.last_updated.isoformat()
                })

        # Get recent system issues (e.g., low confidence classifications)
        recent_collections = WasteCollection.query.filter(
            WasteCollection.confidence < 0.5
        ).limit(5).all()

        for collection in recent_collections:
            notifications.append({
                'type': 'info',
                'title': 'Low Confidence Detection',
                'message': f'Item classified as {collection.item_type} with {collection.confidence*100:.1f}% confidence',
                'timestamp': collection.timestamp.isoformat()
            })
            
        # Add notification if Arduino is disconnected
        if not arduino_handler.connected:
             notifications.append({
                'type': 'error',
                'title': 'Arduino Disconnected',
                'message': 'The physical waste management unit is offline. Please check the connection.',
                'timestamp': datetime.utcnow().isoformat()
            })

        # Sort by timestamp (newest first)
        notifications.sort(key=lambda x: x['timestamp'], reverse=True)

        return jsonify({
            'notifications': notifications[:10],  # Limit to 10 most recent
            'unread_count': len([n for n in notifications if n['type'] == 'warning' or n['type'] == 'error'])
        })

    except Exception as e:
        print(f"Error getting mobile notifications: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mobile/offline_data')
def get_offline_data():
    """Get data for offline functionality."""
    try:
        # Get recent collections for offline viewing
        recent_collections = WasteCollection.query.order_by(
            WasteCollection.timestamp.desc()
        ).limit(50).all()

        # Get container status
        containers = ContainerStatus.query.all()

        # Get today's stats
        today = datetime.now().date()
        today_stats = SystemStats.query.filter_by(date=today).first()

        return jsonify({
            'offline_data': {
                'last_updated': datetime.utcnow().isoformat(),
                'recent_collections': [{
                    'item_type': c.item_type,
                    'confidence': c.confidence,
                    'container_id': c.container_id,
                    'method': c.method,
                    'timestamp': c.timestamp.isoformat()
                } for c in recent_collections],
                'containers': [{
                    'container_id': c.container_id,
                    'fill_level': c.fill_level,
                    'status': c.status,
                    'last_updated': c.last_updated.isoformat()
                } for c in containers],
                'today_stats': {
                    'total_collections': today_stats.total_collections if today_stats else 0,
                    'biodegradable_count': today_stats.biodegradable_count if today_stats else 0,
                    'non_biodegradable_count': today_stats.non_biodegradable_count if today_stats else 0
                }
            }
        })

    except Exception as e:
        print(f"Error getting offline data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_data', methods=['POST'])
def clear_data():
    """Clear all database data and reset the system."""
    try:
        # Confirm action (should be called with confirmation from frontend)
        data = request.get_json() or {}
        confirm = data.get('confirm', False)

        if not confirm:
            return jsonify({'error': 'Confirmation required'}), 400

        print("🗑️ Clearing all database data...")

        # Clear all tables
        WasteCollection.query.delete()
        ContainerStatus.query.delete()
        SystemStats.query.delete()

        # Reset container fill levels to 0
        # Create default containers with empty status
        bio_container = ContainerStatus(
            container_id=1,
            fill_level=0.0,
            status='empty',
            capacity=100.0,
            current_weight=0.0,
            last_updated=datetime.utcnow()
        )
        non_bio_container = ContainerStatus(
            container_id=2,
            fill_level=0.0,
            status='empty',
            capacity=100.0,
            current_weight=0.0,
            last_updated=datetime.utcnow()
        )

        db.session.add(bio_container)
        db.session.add(non_bio_container)

        db.session.commit()

        print("✅ Database cleared and containers reset")

        # Emit real-time update to notify clients
        socketio.emit('data_cleared', {
            'message': 'All data cleared and system reset',
            'timestamp': datetime.utcnow().isoformat()
        })

        return jsonify({
            'success': True,
            'message': 'All data cleared and system reset successfully'
        })

    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        db.session.rollback()
        return jsonify({'error': f'Failed to clear data: {str(e)}'}), 500

# SocketIO events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'msg': 'Connected to waste management system'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_camera')
def handle_start_camera():
    global camera_active
    camera_active = True
    emit('camera_status', {'active': True})

@socketio.on('stop_camera')
def handle_stop_camera():
    global camera_active
    camera_active = False
    emit('camera_status', {'active': False})

def cleanup_system():
    """Clean up system resources using the handler's cleanup function."""
    print("🧹 Cleaning up system resources...")
    # The cleanup_arduino function in iot_handler stops the thread and closes the serial port.
    cleanup_arduino()
    print("✅ System cleanup completed")

def status_update_thread():
    """Background thread to emit periodic status updates."""
    print("🔄 Starting status update thread...")

    while True:
        try:
            # Emit current uptime every 5 seconds
            uptime = (datetime.now() - system_start_time).total_seconds()
            socketio.emit('uptime_update', {
                'uptime_seconds': uptime,
                'uptime_formatted': str(timedelta(seconds=int(uptime))),
                'arduino_connected': arduino_handler.connected
            })

            # Update today's stats with current uptime
            with app.app_context():
                today = datetime.now().date()
                stats = SystemStats.query.filter_by(date=today).first()
                if not stats:
                    stats = SystemStats(date=today)
                    db.session.add(stats)
                stats.uptime_hours = uptime / 3600  # Convert to hours
                db.session.commit()

            time.sleep(5)  # Update every 5 seconds

        except Exception as e:
            print(f"❌ Error in status update thread: {e}")
            time.sleep(10)  # Wait longer on error

def initialize_system():
    """Initialize all system components."""
    global model_interpreter

    print("🚀 Initializing waste management system...")

    # Initialize Hugging Face API (always available)
    print("✅ Hugging Face API available")

    # Initialize TFLite model
    model_interpreter = initialize_tflite_model()
    if model_interpreter:
        print("✅ Local TFLite model initialized")
    else:
        print("❌ Local TFLite model not available")

    # Initialize Arduino connection using improved handler (will auto-detect if default fails)
    if initialize_arduino():
        print("✅ Arduino connected and monitoring started.")
    else:
        print("❌ Arduino not connected. System will attempt reconnection on next command.")

    # Start status update thread
    status_thread = threading.Thread(target=status_update_thread, daemon=True)
    status_thread.start()
    print("✅ Status update thread started")

    # Register cleanup function
    atexit.register(cleanup_system)

if __name__ == '__main__':
    # Initialize system components
    initialize_system()

    # Start Flask-SocketIO server
    print("Starting web server...")
    # CRITICAL FIX: Disabling the reloader prevents the parent process from holding the COM port lock 
    # when the child process (the restarted app) tries to take it over.
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True) 
