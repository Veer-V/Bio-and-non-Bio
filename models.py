from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class WasteCollection(db.Model):
    """Model for waste collection records."""
    __tablename__ = 'waste_collection'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    item_type = db.Column(db.String(50), nullable=False)  # biodegradable/non-biodegradable
    confidence = db.Column(db.Float, nullable=False)
    container_id = db.Column(db.Integer, nullable=False)
    image_path = db.Column(db.String(200))
    method = db.Column(db.String(20), nullable=False)  # 'huggingface_api' or 'local_model'
    quantity = db.Column(db.Float, default=1.0)  # estimated quantity/weight

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'item_type': self.item_type,
            'confidence': self.confidence,
            'container_id': self.container_id,
            'image_path': self.image_path,
            'method': self.method,
            'quantity': self.quantity
        }

class ContainerStatus(db.Model):
    """Model for container status tracking."""
    __tablename__ = 'container_status'

    id = db.Column(db.Integer, primary_key=True)
    container_id = db.Column(db.Integer, nullable=False, unique=True)
    fill_level = db.Column(db.Float, default=0.0)  # percentage (0-100)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='empty')  # 'empty', 'filling', 'full', 'overflow'
    capacity = db.Column(db.Float, default=100.0)  # container capacity in kg
    current_weight = db.Column(db.Float, default=0.0)

    def to_dict(self):
        return {
            'id': self.id,
            'container_id': self.container_id,
            'fill_level': self.fill_level,
            'last_updated': self.last_updated.isoformat(),
            'status': self.status,
            'capacity': self.capacity,
            'current_weight': self.current_weight
        }

class SystemStats(db.Model):
    """Model for daily system statistics."""
    __tablename__ = 'system_stats'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, unique=True)
    total_collections = db.Column(db.Integer, default=0)
    biodegradable_count = db.Column(db.Integer, default=0)
    non_biodegradable_count = db.Column(db.Integer, default=0)
    uptime_hours = db.Column(db.Float, default=0)
    total_weight_biodegradable = db.Column(db.Float, default=0.0)
    total_weight_non_biodegradable = db.Column(db.Float, default=0.0)

    def to_dict(self):
        return {
            'id': self.id,
            'date': self.date.isoformat(),
            'total_collections': self.total_collections,
            'biodegradable_count': self.biodegradable_count,
            'non_biodegradable_count': self.non_biodegradable_count,
            'uptime_hours': self.uptime_hours,
            'total_weight_biodegradable': self.total_weight_biodegradable,
            'total_weight_non_biodegradable': self.total_weight_non_biodegradable
        }

class UserSession(db.Model):
    """Model for user sessions (for mobile app)."""
    __tablename__ = 'user_sessions'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), unique=True, nullable=False)
    user_agent = db.Column(db.String(200))
    ip_address = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)
    is_mobile = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'user_agent': self.user_agent,
            'ip_address': self.ip_address,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'is_mobile': self.is_mobile
        }

class PredictionLog(db.Model):
    """Model for detailed prediction logs."""
    __tablename__ = 'prediction_logs'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_hash = db.Column(db.String(64))  # SHA-256 hash of image
    cloud_prediction = db.Column(db.String(50))
    cloud_confidence = db.Column(db.Float)
    local_prediction = db.Column(db.String(50))
    local_confidence = db.Column(db.Float)
    final_prediction = db.Column(db.String(50))
    final_confidence = db.Column(db.Float)
    processing_time = db.Column(db.Float)  # in seconds
    success = db.Column(db.Boolean, default=True)

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'image_hash': self.image_hash,
            'cloud_prediction': self.cloud_prediction,
            'cloud_confidence': self.cloud_confidence,
            'local_prediction': self.local_prediction,
            'local_confidence': self.local_confidence,
            'final_prediction': self.final_prediction,
            'final_confidence': self.final_confidence,
            'processing_time': self.processing_time,
            'success': self.success
        }
