import serial
import time
import threading
from datetime import datetime
from flask import current_app
# Placeholder imports for running without full project structure. 
# You should use your actual models and db imports.
try:
    from models import ContainerStatus, db
    from flask_socketio import emit # Needed for real-time updates
except ImportError:
    print("Warning: Missing real 'models' or 'socketio' imports. Using mock classes.")
    
    # --- Mock Implementations for missing project components ---
    class MockContainerStatus:
        def __init__(self, container_id, fill_level=0.0, status='empty', capacity=100.0, current_weight=0.0):
            self.container_id = container_id
            self.fill_level = fill_level
            self.status = status
            self.capacity = capacity
            self.current_weight = current_weight
            self.last_updated = datetime.utcnow()
        def to_dict(self):
            return {
                'container_id': self.container_id,
                'fill_level': self.fill_level,
                'status': self.status,
                'last_updated': self.last_updated.isoformat()
            }
        @classmethod
        def query(cls):
            # Mocking query functionality for demonstration
            class MockQuery:
                def __init__(self):
                    self.data = {1: MockContainerStatus(1, status='empty'), 2: MockContainerStatus(2, status='empty')}
                def filter_by(self, container_id):
                    class MockFilter:
                        def __init__(self, data):
                            self.data = data
                        def first(self):
                            return self.data.get(container_id)
                    return MockFilter(self.data)
                def all(self):
                    return list(self.data.values())
                def delete(self):
                    pass
            return MockQuery()
    ContainerStatus = MockContainerStatus

    class MockDB:
        def __init__(self):
            self.session = self
        def add(self, obj):
            pass
        def commit(self):
            pass
        def rollback(self):
            pass
    db = MockDB()
    
    class MockSocketIO:
        def emit(self, event, data):
            # Suppress excessive logging if the app is not running with SocketIO
            # print(f"SOCKETIO EMIT: {event} - {data}")
            pass
    socketio_mock = MockSocketIO()
    def emit(*args, **kwargs):
        if current_app: # Only emit if Flask context exists (i.e., when running app.py)
            socketio_mock.emit(*args, **kwargs)
    # --------------------------------------------------------------------------------------


class ArduinoHandler:
    """Handler for Arduino communication and sensor data processing."""

    # Using COM6 as default initial port for Windows, can be changed/overridden by auto-detect
    def __init__(self, port='COM6', baudrate=9600): 
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.connected = False
        self.running = False
        self.data_thread = None
        self.last_container_update = {}
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        
        # Expanded list of common COM ports for Windows/Linux
        self.com_ports = [f'COM{i}' for i in range(1, 21)] + ['/dev/ttyACM0', '/dev/ttyUSB0']

    def connect(self):
        """Connect to Arduino, attempting the configured port first, then auto-detect."""
        if self.port:
            if self._connect_to_port(self.port):
                return True
            print(f"⚠️ Connection failed on configured port {self.port}. Starting auto-detection...")
        
        return self._auto_detect_port()

    def _connect_to_port(self, port):
        """Connect to a specific port."""
        if self.serial_connection and self.serial_connection.is_open:
            self.disconnect() # Close existing connection if open

        try:
            print(f"🔌 Attempting connection to {port}...")
            # Use a slightly longer timeout for robust connection
            self.serial_connection = serial.Serial(port, self.baudrate, timeout=3)
            time.sleep(2)  # Wait for Arduino to initialize

            # Test connection by sending a ping
            print(f"📡 Sending PING to {port}...")
            self.serial_connection.write(b'PING\n')
            time.sleep(1)

            if self.serial_connection.in_waiting > 0:
                # Read until newline, but discard any preceding partial data
                response = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                print(f"📥 Response from {port}: '{response}'")
                
                # The Arduino sent "DISTANCE:175.3" in your log, which is a successful read.
                if response: 
                    self.connected = True
                    self.port = port
                    self.connection_attempts = 0 # Reset attempts on success
                    print(f"✅ Arduino connected successfully on {port}!")
                    return True
                else:
                    print(f"📡 No meaningful PING response from {port}, but connection established.")
            
            # If no response was received (in_waiting == 0) after PING, proceed anyway
            self.connected = True
            self.port = port
            self.connection_attempts = 0 # Reset attempts on success
            print(f"✅ Arduino connected successfully on {port} (assuming readiness)!")
            return True

        except serial.SerialException as e:
            error_msg = str(e).lower()
            if "permissionerror" in error_msg or "access is denied" in error_msg:
                print(f"⚠️ Port {port}: Permission denied (port in use by another program)")
            elif "file not found" in error_msg or "device not recognized" in error_msg or "no such file" in error_msg:
                print(f"⚠️ Port {port}: Device not found")
            else:
                print(f"❌ Serial error on {port}: {e}")
            self.serial_connection = None # Ensure connection is None on failure
            return False
        except Exception as e:
            print(f"❌ Unexpected error connecting to {port}: {e}")
            self.serial_connection = None
            return False

    def _auto_detect_port(self):
        """Auto-detect Arduino port by trying common ports."""
        
        print("🔍 Auto-detecting Arduino connection...")
        print(f"   Scanning ports: {', '.join(self.com_ports)}")

        for port in self.com_ports:
            # Need to create a temporary handler for each attempt since the main one might be in an error state
            try:
                temp_serial = serial.Serial(port, self.baudrate, timeout=1)
                time.sleep(2)
                temp_serial.write(b'PING\n')
                time.sleep(1)
                
                if temp_serial.is_open:
                    # If we can open it and it doesn't immediately error, consider it a candidate.
                    # We transfer this successful connection to the main handler.
                    self.serial_connection = temp_serial
                    self.connected = True
                    self.port = port
                    self.connection_attempts = 0
                    print(f"✅ Arduino connected successfully via auto-detect on {port}!")
                    return True
            except serial.SerialException:
                pass # Expected for non-existent or busy ports

        print("❌ Failed to auto-detect Arduino on any COM port.")
        return False

    def disconnect(self):
        """Disconnect from Arduino and ensure the port is released."""
        self.running = False
        if self.serial_connection and self.serial_connection.is_open:
            try:
                # Send a final signal to the Arduino
                self.serial_connection.write(b'CLOSE\n')
                time.sleep(0.1)
            except:
                pass # Ignore error if port is already broken

            try:
                self.serial_connection.close()
                print("🔌 Serial port closed and released.")
            except Exception as e:
                print(f"⚠️ Error closing serial port: {e}")
            
            self.serial_connection = None # CRITICAL: Ensure the reference is cleared
        self.connected = False
        print("🔌 Disconnected from Arduino")

    def send_command(self, command):
        """Send command to Arduino."""
        if self.connected and self.serial_connection and self.serial_connection.is_open:
            try:
                # Ensure command ends with a newline character for the Arduino sketch
                if not command.endswith('\n'):
                    command += '\n'
                
                self.serial_connection.write(command.encode())
                print(f"📤 Sent to Arduino: {command.strip()}")
                return True
            except Exception as e:
                print(f"❌ Error sending command to Arduino: {e}")
                # Attempt to handle a connection drop during a send operation
                self._handle_connection_error()
                return False
        else:
            print("❌ Arduino not connected or port not open. Attempting reconnection...")
            if self._handle_connection_error():
                # Retry send command after successful reconnection
                return self.send_command(command.strip('\n'))
            return False

    def read_data(self):
        """Read data from Arduino with improved error handling."""
        if not self.connected or not self.serial_connection or not self.serial_connection.is_open:
            return None, False

        try:
            # Check if port is still available and has data
            if self.serial_connection.in_waiting > 0:
                # Read all available data and then split it by newline
                data_bytes = self.serial_connection.read(self.serial_connection.in_waiting)
                data_str = data_bytes.decode('utf-8', errors='ignore').strip()
                
                # Split multiple lines if received
                lines = [line.strip() for line in data_str.split('\n') if line.strip()]
                
                # Process all lines received, but return only the first for the current iteration
                if lines:
                    for line in lines[1:]:
                        self._process_data(line) # Recursively process remaining lines
                    return lines[0], False 
                
            return None, False
        
        except serial.SerialException as e:
            print(f"⚠️ Serial port error detected: {e}")
            self._handle_connection_error()
            return None, True
        except UnicodeDecodeError as e:
            print(f"⚠️ Data decoding error: {e}")
            return None, True
        except Exception as e:
            print(f"❌ Unexpected error reading from Arduino: {e}")
            return None, True

    def start_monitoring(self):
        """Start monitoring Arduino data."""
        if not self.connected:
            print("❌ Cannot start monitoring: Arduino not connected")
            return False

        self.running = True
        self.data_thread = threading.Thread(target=self._monitor_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        print("📊 Started Arduino monitoring")
        return True

    def stop_monitoring(self):
        """Stop monitoring Arduino data."""
        self.running = False
        if self.data_thread:
            # Send a stop signal to the thread and wait briefly
            if threading.current_thread() != self.data_thread:
                 self.data_thread.join(timeout=1)
        print("⏹️ Stopped Arduino monitoring")

    def _monitor_loop(self):
        """Main monitoring loop with connection health checks."""
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.running:
            # read_data internally handles connection errors via _handle_connection_error
            data, is_error = self.read_data()
            
            if not self.connected:
                print("❌ Monitoring stopped: Connection lost and failed to reconnect.")
                break 

            if data:
                consecutive_errors = 0 
                self._process_data(data)
            elif is_error:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"⚠️ {consecutive_errors} consecutive read failures detected. Checking health...")
                    self._handle_connection_error() 
                    consecutive_errors = 0 
            
            time.sleep(0.1)

    def _process_data(self, data):
        """Process incoming data from Arduino."""
        try:
            print(f"📥 Received from Arduino: {data}")
            
            if data.startswith('DISTANCE:') or data.startswith('SENSOR:DISTANCE'):
                # Process distance data for background monitoring/logging, but don't clutter system logs excessively
                if self.running: # Only log distance when actively monitoring
                    self._process_sensor_data(data) 
                else: # Only log distance once if not running (e.g. during PING response)
                    print(f"🔬 Sensor Data (Initial read): {data}")
                return

            elif data.startswith('CONTAINER:'):
                self._process_container_data(data)
            elif data.startswith('SENSOR:'):
                self._process_sensor_data(data)
            elif data.startswith('STATUS:'):
                self._process_status_data(data)
            elif data.startswith('ERROR:'):
                print(f"🚨 Arduino Error: {data}")
            elif data.startswith('ACK:'):
                 print(f"✅ Arduino Acknowledgment: {data}")
            else:
                print(f"📥 Received Unhandled Data: {data}")

        except Exception as e:
            print(f"❌ Error processing Arduino data: {e}")

    def _process_container_data(self, data):
        """Process container status data (e.g., CONTAINER:1:75.5:FULL)."""
        try:
            parts = data.split(':')
            if len(parts) >= 4:
                container_id = int(parts[1])
                fill_level = float(parts[2])
                status = parts[3]

                # Update database (requires Flask app_context)
                if current_app:
                    with current_app.app_context():
                        container = ContainerStatus.query.filter_by(container_id=container_id).first()
                        if not container:
                            container = ContainerStatus(container_id=container_id)
                            db.session.add(container)

                        container.fill_level = fill_level
                        container.status = status
                        container.last_updated = datetime.utcnow()
                        db.session.commit()

                        print(f"📦 Container {container_id}: {fill_level}% - {status}")

                        # Emit real-time update via SocketIO
                        emit('container_update', {
                            'container_id': container_id,
                            'fill_level': fill_level,
                            'status': status,
                            'timestamp': container.last_updated.isoformat()
                        }, namespace='/', broadcast=True)

        except Exception as e:
            print(f"❌ Error processing container data: {e}")

    def _process_sensor_data(self, data):
        """Process sensor readings (e.g., SENSOR:DISTANCE:1:25.5 or DISTANCE:25.5)."""
        try:
            parts = data.split(':')
            sensor_type = 'DISTANCE'
            container_id = 1 # Default to container 1 if not specified
            value = None

            if parts[0] == 'DISTANCE' and len(parts) >= 2:
                 value = float(parts[1])
            elif parts[0] == 'SENSOR' and len(parts) >= 4:
                sensor_type = parts[1]
                container_id = int(parts[2])
                value = float(parts[3])
            
            if sensor_type == 'DISTANCE' and value is not None:
                self._update_container_distance(container_id, value)
            elif sensor_type == 'WEIGHT' and value is not None:
                 self._update_container_weight(container_id, value)

        except Exception as e:
            print(f"❌ Error processing sensor data: {e}")


    def _update_container_weight(self, container_id, weight):
        """Update container weight."""
        try:
            if current_app:
                with current_app.app_context():
                    container = ContainerStatus.query.filter_by(container_id=container_id).first()
                    if container:
                        container.current_weight = weight
                        # Calculate fill level based on weight if needed
                        if hasattr(container, 'capacity') and container.capacity > 0:
                            container.fill_level = min(100, (weight / container.capacity) * 100)
                            container.status = self._get_status_from_fill_level(container.fill_level)
                        db.session.commit()

        except Exception as e:
            print(f"❌ Error updating container weight: {e}")

    def _update_container_distance(self, container_id, distance):
        """Update container fill level based on distance sensor."""
        try:
            # Assuming distance sensor measures from top
            max_distance = 30  # cm (empty)
            min_distance = 5   # cm (full)

            if distance <= min_distance:
                fill_level = 100
                status = 'full'
            elif distance >= max_distance:
                fill_level = 0
                status = 'empty'
            else:
                fill_level = ((max_distance - distance) / (max_distance - min_distance)) * 100
                status = self._get_status_from_fill_level(fill_level)
            
            if current_app:
                with current_app.app_context():
                    container = ContainerStatus.query.filter_by(container_id=container_id).first()
                    if not container:
                        # Use mock or actual model initializer
                        container = ContainerStatus(container_id=container_id)
                        db.session.add(container)

                    # Only update if the fill level reading is significant
                    if abs(container.fill_level - fill_level) > 1.0 or container.status != status:
                        container.fill_level = fill_level
                        container.status = status
                        container.last_updated = datetime.utcnow()
                        db.session.commit()
                        print(f"🔬 Container {container_id} Distance Update: {fill_level:.1f}% ({status})")

        except Exception as e:
            print(f"❌ Error updating container distance: {e}")
            
    def _get_status_from_fill_level(self, fill_level):
        """Get status string from fill level percentage."""
        if fill_level >= 95:
            return 'full'
        elif fill_level >= 70:
            return 'filling'
        elif fill_level >= 30:
            return 'moderate'
        else:
            return 'empty'

    def _process_status_data(self, data):
        """Process system status from Arduino."""
        try:
            # Expected format: STATUS:READY or STATUS:BUSY
            status = data.split(':')[1]
            print(f"🔄 Arduino Status: {status}")

        except Exception as e:
            print(f"❌ Error processing status data: {e}")

    def _handle_connection_error(self):
        """Handle connection errors by attempting reconnection."""
        # ... (Original _handle_connection_error logic)
        if self.connection_attempts >= self.max_connection_attempts:
            print(f"⚠️ Maximum reconnection attempts ({self.max_connection_attempts}) reached")
            self.connected = False 
            print("   Please check Arduino connection and restart the application")
            return False

        self.connection_attempts += 1
        print(f"🔄 Reconnection attempt {self.connection_attempts}/{self.max_connection_attempts}")

        # Disconnect current connection
        self.disconnect()

        time.sleep(2)

        if self.connect():
            print("✅ Successfully reconnected to Arduino!")
            return True
        else:
            print(f"❌ Reconnection attempt {self.connection_attempts} failed")
            return False

    def _check_connection_health(self):
        """Check if the current connection is healthy by sending PING."""
        # ... (Original _check_connection_health logic)
        if not self.connected or not self.serial_connection or not self.serial_connection.is_open:
            return False

        try:
            self.serial_connection.write(b'PING\n')
            time.sleep(0.5)
            if self.serial_connection.in_waiting > 0:
                response = self.serial_connection.readline().decode('utf-8').strip()
                if len(response) > 0:
                    print("✅ Connection health check: Successful PING response.")
                    return True
            
            if self.serial_connection.is_open and self.serial_connection.readable:
                return True
                
        except Exception as e:
            print(f"⚠️ Connection health check failed: {e}")
            return False


# Global instance
arduino_handler = ArduinoHandler(port='COM6') 

def initialize_arduino(port=None):
    """Initialize Arduino connection and start monitoring."""
    global arduino_handler
    
    # Check if a different port is requested
    if port and port != arduino_handler.port:
        arduino_handler.disconnect()
        arduino_handler = ArduinoHandler(port=port)
    
    if arduino_handler.connect():
        arduino_handler.start_monitoring()
        return True
    return False

def cleanup_arduino():
    """Cleanup Arduino connection."""
    # Ensure monitoring thread is stopped before disconnecting
    arduino_handler.stop_monitoring() 
    arduino_handler.disconnect()
