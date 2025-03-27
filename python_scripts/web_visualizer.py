import time
from flask_socketio import SocketIO

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
            # Check if get_led_colors exists, if not, use get_colors directly
            if hasattr(led_visualizer, 'get_led_colors'):
                led_colors = led_visualizer.get_led_colors()
            else:
                led_colors = led_visualizer.get_colors()
            
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
                    # Debug - log a sample of the color values less frequently
                    if frame_count % 120 == 0:
                        sample_colors = led_colors[:3] if led_colors else []
                        logger.info(f"Sample LED colors: {sample_colors}")
                    
                    # Send with retry
                    success = arduino.send_data(led_colors)
                    if not success and frame_count % 100 == 0:
                        logger.warning("Failed to send data to Arduino")
                except Exception as e:
                    logger.error(f"Error sending data to Arduino: {str(e)}")
            
            # Send data to client for preview
            frame_count += 1
            now = time.time()
            if now - last_update_time >= 0.2:  # Update client at 5 Hz instead of 10 Hz
                fps = frame_count / (now - last_update_time)
                
                # Prepare data for the client - limit to 30 LEDs for preview
                if led_colors and isinstance(led_colors[0], tuple):
                    # Convert tuples to flat array for web client
                    flat_colors = []
                    for r, g, b in led_colors[:30]:  # Limit to first 30 LEDs
                        flat_colors.extend([r, g, b])
                    client_colors = flat_colors
                else:
                    # Already flat, just limit size
                    client_colors = led_colors[:30*3]
                
                buffer_fullness = 0
                target_fps = 0
                arduino_fps = 0
                if arduino:
                    buffer_fullness = arduino.buffer_fullness
                    target_fps = arduino.target_fps
                    arduino_fps = arduino.arduino_fps
                
                # Log every 60 frames instead of 30 to reduce logging
                if frame_count % 60 == 0:
                    logger.info(f"Visualization - FPS: {fps:.1f}, Arduino FPS: {arduino_fps:.1f}, Target: {target_fps:.1f}, Buffer: {buffer_fullness*100:.1f}%")
                
                socketio.emit('visualization_data', {
                    'led_colors': client_colors,
                    'beat': 1 if beat_detected else 0,
                    'fps': round(fps, 1),
                    'target_fps': round(target_fps, 1),
                    'arduino_fps': round(arduino_fps, 1),
                    'buffer': round(buffer_fullness * 100, 1)
                })
                last_update_time = now
                frame_count = 0
            
            # Minimal sleep to avoid 100% CPU usage and give Arduino a breather
            if not stop_event.is_set():
                time.sleep(0.002)  # Increased from 0.001 to 0.002
    
    except Exception as e:
        logger.exception(f"Error in visualization loop: {e}")
    finally:
        logger.info("Visualization loop stopped") 