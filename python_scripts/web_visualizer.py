#!/usr/bin/env python3
"""
Web Visualizer Server
Provides a web interface to preview LED strip visualizations
"""
import sys
import time
import json
import threading
import argparse
import logging
import numpy as np
from flask import Flask, render_template, send_from_directory, jsonify, request
from flask_socketio import SocketIO, emit
from serial_manager import ArduinoManager

try:
    from python_scripts.audio_visualizer import AudioCapture, AudioAnalyzer
    from python_scripts.audio_visualizations import create_visualizer, VISUALIZATIONS
    from python_scripts.audio_test import MockAudioGenerator
    import sounddevice as sd
except ImportError:
    from audio_visualizer import AudioCapture, AudioAnalyzer
    from audio_visualizations import create_visualizer, VISUALIZATIONS
    from audio_test import MockAudioGenerator
    import sounddevice as sd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('web_visualizer')

# Initialize Flask app
app = Flask(__name__, template_folder='web_templates', static_folder='web_static')
app.config['SECRET_KEY'] = 'lightingsimulation'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
visualizer_thread = None
stop_event = threading.Event()
arduino = None
audio_capture = None
audio_analyzer = None
led_visualizer = None

@app.route('/')
def index():
    """Main page route"""
    return render_template('led_visualizer.html')

@app.route('/api/devices')
def get_devices():
    """API route to get available audio devices"""
    try:
        devices = sd.query_devices()
        device_list = []
        
        for i, device in enumerate(devices):
            # Include both input and output devices that have channels
            if device['max_input_channels'] > 0 or device['max_output_channels'] > 0:
                device_list.append({
                    'id': i,
                    'name': device['name'],
                    'inputs': device['max_input_channels'],
                    'outputs': device['max_output_channels'],
                    'sample_rate': device['default_samplerate']
                })
        
        logger.info(f"Found {len(device_list)} audio devices")
        return jsonify({'success': True, 'devices': device_list})
    except Exception as e:
        logger.error(f"Error getting audio devices: {e}")
        return jsonify({'success': False, 'error': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected")
    emit('status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected")
    # Stop visualization if running
    stop_visualization()

@socketio.on('start_visualization')
def handle_start_visualization(data):
    """Handle starting visualization"""
    global visualizer_thread, audio_capture, audio_analyzer, led_visualizer, arduino, stop_event
    
    # Stop any existing visualization
    stop_visualization()
    
    # Reset stop event
    stop_event.clear()
    
    try:
        # Get settings from client
        device_id = int(data.get('device_id', sd.default.device[0]))
        device_name = data.get('device_name', 'Default Device')
        visualization_type = data.get('visualization_type', 'spectrum')
        arduino_port = data.get('arduino_port', '/dev/tty.usbserial-110')
        
        logger.info(f"Starting visualization: {visualization_type} on device {device_name} ({device_id})")
        
        # Mock Arduino if needed for testing
        mock_arduino = data.get('mock_arduino', False)
        if mock_arduino:
            logger.info("Using mock Arduino mode")
            arduino = None
        else:
            # Initialize Arduino connection
            arduino = ArduinoManager(port=arduino_port)
            arduino_connected = arduino.connect()
            
            if not arduino_connected:
                logger.error(f"Failed to connect to Arduino on port {arduino_port}")
                emit('error', {'message': f'Failed to connect to Arduino on port {arduino_port}'})
                return
        
        # Check if we should use mock audio
        use_mock_audio = data.get('use_mock_audio', False)
        if use_mock_audio:
            logger.info("Using mock audio generator")
            audio_capture = MockAudioGenerator(sample_rate=48000, buffer_size=1024)
            capture_started = True
        else:
            # Initialize audio capture with chosen device
            audio_capture = AudioCapture(device_name=device_name, device_id=device_id, channels=1)
            capture_started = audio_capture.start()
        
        if not capture_started:
            logger.error(f"Failed to start audio capture from device: {device_name}")
            emit('error', {'message': f'Failed to start audio capture from device: {device_name}'})
            return
        
        # Initialize audio analyzer
        audio_analyzer = AudioAnalyzer(sample_rate=48000)
        
        # Initialize LED visualizer
        led_visualizer = create_visualizer(visualization_type)
        
        # Start visualization thread
        visualizer_thread = threading.Thread(target=run_visualization)
        visualizer_thread.daemon = True
        visualizer_thread.start()
        
        emit('visualization_started', {
            'visualization_type': visualization_type,
            'device': device_name
        })
        
    except Exception as e:
        logger.exception(f"Error starting visualization: {e}")
        emit('error', {'message': f'Error starting visualization: {str(e)}'})
        stop_visualization()

@socketio.on('stop_visualization')
def handle_stop_visualization(data=None):
    """Handle stopping visualization"""
    stop_res = stop_visualization()
    if stop_res:
        emit('visualization_stopped', {})
    else:
        emit('error', {'message': 'Failed to stop visualization'})

def stop_visualization():
    """Stop the visualization thread and clean up resources"""
    global visualizer_thread, audio_capture, audio_analyzer, led_visualizer, arduino, stop_event
    
    try:
        logger.info("Stopping visualization")
        
        # Signal thread to stop
        stop_event.set()
        
        # Wait for thread to finish
        if visualizer_thread and visualizer_thread.is_alive():
            visualizer_thread.join(timeout=2.0)
        
        # Clean up resources
        if audio_capture:
            audio_capture.stop()
            audio_capture = None
            
        if arduino:
            # Turn off all LEDs
            try:
                arduino.send_data([0, 0, 0] * 60)
                arduino.disconnect()
            except Exception as e:
                logger.error(f"Error while stopping Arduino: {e}")
            finally:
                arduino = None
            
        audio_analyzer = None
        led_visualizer = None
        visualizer_thread = None
        
        logger.info("Visualization stopped and resources cleaned up")
        return True
        
    except Exception as e:
        logger.exception(f"Error stopping visualization: {e}")
        return False

def run_visualization():
    """Run the visualization loop"""
    last_update_time = time.time()
    frame_count = 0
    
    try:
        logger.info("Starting visualization loop")
        while not stop_event.is_set():
            # Get audio data
            audio_data = audio_capture.get_audio_data()
            
            if audio_data is None or len(audio_data) == 0:
                time.sleep(0.01)
                continue
                
            # Analyze audio
            fft_data = audio_analyzer.compute_fft(audio_data)
            beat_detected = audio_analyzer.detect_beat(audio_data)
            frequency_bands = audio_analyzer.get_frequency_bands(fft_data)
            
            # Update visualizer
            led_visualizer.update(
                audio_data=audio_data,
                fft_data=fft_data,
                beat_detected=beat_detected,
                frequency_bands=frequency_bands
            )
            
            # Get LED colors
            led_colors = led_visualizer.get_led_colors()
            
            # Send to Arduino
            if arduino and arduino.is_connected:
                arduino.send_data(led_colors)
            
            # Send data to client for preview
            frame_count += 1
            now = time.time()
            if now - last_update_time >= 0.1:  # Update client at 10 Hz
                fps = frame_count / (now - last_update_time)
                socketio.emit('visualization_data', {
                    'led_colors': led_colors[:60*3],  # Limit data size
                    'beat': beat_detected,
                    'fps': round(fps, 1)
                })
                last_update_time = now
                frame_count = 0
                
            # Prevent CPU overload
            time.sleep(0.01)
            
    except Exception as e:
        logger.exception(f"Error in visualization thread: {e}")
        socketio.emit('error', {'message': f'Visualization error: {str(e)}'})
        stop_visualization()

def main():
    """Run the web visualizer server"""
    parser = argparse.ArgumentParser(description='LED Web Visualizer')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    
    args = parser.parse_args()
    
    if args.list_devices:
        devices = sd.query_devices()
        print("\nAvailable audio devices:")
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']}")
            if device.get('max_input_channels', 0) > 0:
                print(f"  Input channels: {device['max_input_channels']}")
            if device.get('max_output_channels', 0) > 0:
                print(f"  Output channels: {device['max_output_channels']}")
        return
    
    print(f"Starting LED Web Visualizer server at http://{args.host}:{args.port}")
    print("Available visualization types:")
    for vis_type in VISUALIZATIONS.keys():
        print(f"  - {vis_type}")
    
    # Start the server
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    main() 