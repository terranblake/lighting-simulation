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
from serial_manager import SerialManager

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
        max_fps = 240
        brightness = float(data.get('brightness', 1.0))
        
        logger.info(f"Starting visualization: {visualization_type} on device {device_name} ({device_id}), max FPS: {max_fps}, brightness: {brightness:.2f}")
        
        # Mock Arduino if needed for testing
        mock_arduino = data.get('mock_arduino', False)
        if mock_arduino:
            logger.info("Using mock Arduino mode")
            arduino = None
        else:
            # Initialize Arduino connection with specified max FPS
            arduino = SerialManager(port=arduino_port, max_fps=max_fps)
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
        
        # Set initial brightness
        led_visualizer.set_param("brightness", brightness)
        
        # Start visualization thread
        visualizer_thread = threading.Thread(target=run_visualization)
        visualizer_thread.daemon = True
        visualizer_thread.start()
        
        emit('visualization_started', {
            'visualization_type': visualization_type,
            'device': device_name,
            'brightness': brightness
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

@socketio.on('update_fps_limit')
def handle_update_fps_limit(data):
    """Handle updating the maximum FPS limit"""
    global arduino
    
    if not arduino:
        emit('error', {'message': 'No active Arduino connection'})
        return
    
    try:
        max_fps = 120
        arduino.max_fps = max_fps
        # arduino.target_fps = min(arduino.target_fps, max_fps)
        arduino.target_fps = 120
        arduino.min_frame_time = 1.0 / arduino.target_fps if arduino.target_fps > 0 else 0
        
        logger.info(f"Updated max FPS limit to {max_fps}")
        emit('fps_limit_updated', {'max_fps': max_fps, 'target_fps': arduino.target_fps})
    except Exception as e:
        logger.error(f"Error updating FPS limit: {str(e)}")
        emit('error', {'message': f'Error updating FPS limit: {str(e)}'})

@socketio.on('update_brightness')
def handle_update_brightness(data):
    """Handle updating the brightness of the LED visualizer"""
    global led_visualizer
    
    if not led_visualizer:
        emit('error', {'message': 'No active visualizer'})
        return
    
    try:
        brightness = float(data.get('brightness', 1.0))
        # Ensure brightness is between 0.0 and 1.0
        brightness = max(0.0, min(1.0, brightness))
        
        # Update visualizer brightness
        led_visualizer.set_param("brightness", brightness)
        
        logger.info(f"Updated brightness to {brightness:.2f}")
        emit('brightness_updated', {'brightness': brightness})
    except Exception as e:
        logger.error(f"Error updating brightness: {str(e)}")
        emit('error', {'message': f'Error updating brightness: {str(e)}'})

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
                arduino.clear_leds()
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
    
    # Statistics for tracking performance
    frames_sent_to_arduino = 0  # Track frames we send to Arduino
    last_sent_stats_time = time.time()
    sent_frame_count = 0
    sent_fps = 0
    
    try:
        logger.info("Starting visualization loop")
        while not stop_event.is_set():
            # Get audio data
            audio_data = audio_capture.get_audio_data()
            
            if audio_data is None or len(audio_data) == 0:
                time.sleep(0.01)
                continue
                
            # Analyze audio - faster with optimized analyzer
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
            
            # Get LED colors - ensure they're in the right format
            led_colors = led_visualizer.get_led_colors()
            
            # Verify format: must be a list of (r,g,b) tuples
            if led_colors and not isinstance(led_colors[0], tuple):
                # Convert if they're not already tuples
                # This ensures compatibility with both old and new visualizers
                rgb_colors = []
                for i in range(0, len(led_colors), 3):
                    if i+2 < len(led_colors):
                        rgb_colors.append((led_colors[i], led_colors[i+1], led_colors[i+2]))
                led_colors = rgb_colors
            
            # Send to Arduino
            if arduino and arduino.is_connected():
                try:
                    # Debug - log a sample of the color values
                    if frame_count % 30 == 0:
                        sample_colors = led_colors[:3] if led_colors else []
                        logger.info(f"Sample LED colors: {sample_colors}")
                    
                    arduino.send_data(led_colors)
                    # Update sent frames count
                    frames_sent_to_arduino += 1
                    sent_frame_count += 1
                except Exception as e:
                    logger.error(f"Error sending data to Arduino: {str(e)}")
            
            # Calculate sent frames FPS
            now = time.time()
            if now - last_sent_stats_time >= 1.0:  # Update stats every second
                sent_fps = sent_frame_count / (now - last_sent_stats_time)
                last_sent_stats_time = now
                sent_frame_count = 0
                logger.info(f"Frames sent to Arduino: {frames_sent_to_arduino}, Current send rate: {sent_fps:.1f} FPS")
            
            # Send data to client for preview
            frame_count += 1
            now = time.time()
            if now - last_update_time >= 0.1:  # Update client at 10 Hz
                fps = frame_count / (now - last_update_time)
                
                # Prepare data for the client
                # The client expects either a flat array [r,g,b,r,g,b,...] or array of tuples
                if led_colors and isinstance(led_colors[0], tuple):
                    # Convert tuples to flat array for web client (historically expected format)
                    flat_colors = []
                    for r, g, b in led_colors[:60]:  # Limit to first 60 LEDs
                        flat_colors.extend([r, g, b])
                    client_colors = flat_colors
                else:
                    # Already flat, just limit size
                    client_colors = led_colors[:60*3]
                
                buffer_fullness = 0
                target_fps = 0
                arduino_fps = 0
                if arduino:
                    buffer_fullness = arduino.buffer_fullness
                    target_fps = arduino.target_fps
                    arduino_fps = arduino.arduino_fps
                    logger.debug(f"Arduino metrics - Buffer: {buffer_fullness*100:.1f}%, Target FPS: {target_fps}, Arduino FPS: {arduino_fps}")
                
                # Log every 30 frames to avoid excessive logging
                if frame_count % 30 == 0:
                    logger.info(f"Sending visualization data - Current FPS: {fps:.1f}, Arduino FPS: {arduino_fps:.1f}, Target: {target_fps:.1f}")
                
                socketio.emit('visualization_data', {
                    'led_colors': client_colors,
                    'beat': 1 if beat_detected else 0,
                    'fps': round(fps, 1),
                    'target_fps': round(target_fps, 1),
                    'arduino_fps': round(arduino_fps, 1),
                    'sent_fps': round(sent_fps, 1),
                    'buffer': round(buffer_fullness * 100, 1)
                })
                last_update_time = now
                frame_count = 0
            
            # Minimal sleep to avoid 100% CPU usage
            if not stop_event.is_set():
                time.sleep(0.001)
                
    except Exception as e:
        logger.exception(f"Error in visualization loop: {e}")
    finally:
        logger.info("Visualization loop stopped")

def main():
    """Run the web visualizer server"""
    parser = argparse.ArgumentParser(description='LED Web Visualizer')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5050,
                        help='Port to run the server on')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    parser.add_argument('--max-fps', type=int, default=120,
                        help='Maximum frames per second to send to Arduino (default: 120)')
    
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
    
    print(f"Starting LED Web Visualizer server at http://0.0.0.0:{args.port}")
    print(f"Maximum FPS: {args.max_fps}")
    # Print available visualization types
    available_types = list(VISUALIZATIONS.keys())
    print(f"Available visualization types: {', '.join(available_types)}")
    
    # Start the server
    socketio.run(app, host=args.host, port=args.port, debug=True, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    main() 