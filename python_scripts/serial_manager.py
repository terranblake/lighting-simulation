import serial
import time
import struct
import threading
import logging

class SerialManager:
    """Manages serial communication with the Arduino controlling the LED strip."""
    
    def __init__(self, port=None, baud_rate=115200, num_leds=60, max_fps=30, auto_reconnect=True):
        self.port = port
        self.baud_rate = baud_rate
        self.num_leds = num_leds
        self.serial_connection = None
        self.auto_reconnect = auto_reconnect
        self.connected = False
        self.lock = threading.Lock()
        self.last_frame_time = 0
        self.frame_count = 0
        self.debug_mode = True
        self.max_fps = max_fps
        self.min_frame_time = 1.0 / max_fps if max_fps > 0 else 0
        self.debug_messages = []
        self.buffer_check_interval = 1.0  # Check buffer status every 1 second
        self.last_buffer_check = 0
        self.buffer_fullness = 0.0  # 0.0 to 1.0
        self.dropped_frames = 0
        
    def connect(self, port=None):
        """Connect to the Arduino via the specified serial port."""
        if port:
            self.port = port
            
        if not self.port:
            raise ValueError("Serial port not specified")
            
        try:
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=1
            )
            
            # Clear any existing data
            self.serial_connection.reset_input_buffer()
            self.serial_connection.reset_output_buffer()
            
            # Wait for Arduino to initialize
            time.sleep(2)
            
            # Check if Arduino is ready by reading any available data
            if self.serial_connection.in_waiting:
                data = self.serial_connection.read(self.serial_connection.in_waiting)
                if b'READY' in data:
                    logging.info("Arduino is ready")
                    
            self.connected = True
            logging.info(f"Connected to Arduino on {self.port}")
            return True
            
        except (serial.SerialException, OSError) as e:
            logging.error(f"Failed to connect to Arduino on {self.port}: {str(e)}")
            self.connected = False
            return False
            
    def disconnect(self):
        """Disconnect from the Arduino."""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            self.connected = False
            logging.info("Disconnected from Arduino")
    
    def is_connected(self):
        """Check if the connection to the Arduino is active."""
        return self.connected and self.serial_connection and self.serial_connection.is_open
    
    def _ensure_connection(self):
        """Ensure that the connection to the Arduino is active."""
        if not self.is_connected():
            if self.auto_reconnect:
                logging.warning("Connection lost. Attempting to reconnect...")
                self.connect()
                if not self.is_connected():
                    raise ConnectionError("Failed to reconnect to Arduino")
            else:
                raise ConnectionError("Not connected to Arduino")
    
    def check_buffer_status(self):
        """Check the Arduino's buffer status and return how full it is (0.0-1.0)."""
        self._ensure_connection()
        
        with self.lock:
            # Send the buffer status command
            self.serial_connection.write(b'B')
            self.serial_connection.flush()
            
            # Wait for response (timeout after 0.1 seconds)
            start_time = time.time()
            while self.serial_connection.in_waiting < 1:
                if time.time() - start_time > 0.1:
                    logging.warning("Timeout waiting for buffer status response")
                    return self.buffer_fullness  # Return last known value
                time.sleep(0.01)
            
            # Read the buffer status byte
            status_byte = self.serial_connection.read(1)
            if status_byte:
                self.buffer_fullness = ord(status_byte) / 255.0
                return self.buffer_fullness
            
            return 0.0
    
    def process_incoming_data(self):
        """Process any incoming data from the Arduino."""
        if not self.is_connected():
            return
            
        try:
            # Check if there's data available
            if not self.serial_connection.in_waiting:
                return
                
            # Read the first byte to determine message type
            message_type = self.serial_connection.read(1)
            
            # Debug message
            if message_type == b'\x02':  # DEBUG_PRINT marker
                message = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                self.debug_messages.append(message)
                if self.debug_mode:
                    logging.debug(f"Arduino: {message}")
                    
            # Acknowledgment
            elif message_type == b'\x01':  # ACK_SUCCESS marker
                # Frame was processed successfully
                pass
                
            # Error
            elif message_type == b'\xFF':  # ACK_ERROR marker
                logging.warning("Arduino reported an error processing the frame")
                
            # Unknown message type
            else:
                # Just read any available data to clear the buffer
                data = self.serial_connection.read(self.serial_connection.in_waiting)
                logging.warning(f"Unknown message type: {message_type.hex()}, data: {data.hex()}")
                
        except Exception as e:
            logging.error(f"Error processing incoming data: {str(e)}")
    
    def send_data(self, led_colors, force=False):
        """
        Send LED color data to the Arduino.
        
        Args:
            led_colors: List of RGB tuples [(r,g,b), ...] for each LED
            force: If True, send the frame regardless of timing constraints
        
        Returns:
            bool: True if data was sent, False if skipped due to rate limiting
        """
        self._ensure_connection()
        
        # Check if we need to limit the frame rate
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        
        # Check buffer status periodically to avoid overwhelming Arduino
        if current_time - self.last_buffer_check > self.buffer_check_interval:
            self.check_buffer_status()
            self.last_buffer_check = current_time
            
            if self.buffer_fullness > 0.7:  # Buffer is getting full
                logging.warning(f"Arduino buffer is at {self.buffer_fullness * 100:.1f}% - slowing down")
                # Skip this frame
                self.dropped_frames += 1
                return False
        
        # Apply frame rate limiting unless forced
        if not force and elapsed < self.min_frame_time:
            self.dropped_frames += 1
            return False
            
        # Prepare the data to send
        if len(led_colors) > self.num_leds:
            led_colors = led_colors[:self.num_leds]
        
        # Ensure all LEDs have values
        while len(led_colors) < self.num_leds:
            led_colors.append((0, 0, 0))
            
        # Debug log a sample of LEDs
        if self.debug_mode and self.frame_count % 100 == 0:
            sample = led_colors[:2]
            logging.debug(f"Sending LED frame {self.frame_count}, sample: {sample}")
        
        with self.lock:
            try:
                # Prepare the data in memory first - more efficient than writing byte by byte
                # Send header: 0xAA followed by number of LEDs
                data = bytearray([0xAA, len(led_colors)])
                
                # Add RGB values for each LED to the buffer
                for r, g, b in led_colors:
                    data.extend([r, g, b])
                
                # Send the entire frame in one write operation
                self.serial_connection.write(data)
                self.serial_connection.flush()
                
                # Process any incoming acknowledgments with timeout
                start_time = time.time()
                
                while time.time() - start_time < 0.1:  # 100ms timeout
                    self.process_incoming_data()
                    
                    # If we've processed all pending data, we're done
                    if not self.serial_connection.in_waiting:
                        break
                    
                    time.sleep(0.001)  # Small sleep to avoid busy waiting
                
                self.frame_count += 1
                self.last_frame_time = current_time
                
                if self.frame_count % 100 == 0:
                    fps = 1.0 / (elapsed if elapsed > 0 else 0.001)
                    logging.debug(f"Sent frame {self.frame_count}, FPS: {fps:.1f}, Buffer: {self.buffer_fullness * 100:.1f}%")
                
                return True
                
            except Exception as e:
                logging.error(f"Error sending data to Arduino: {str(e)}")
                return False
    
    def clear_leds(self):
        """Turn off all LEDs."""
        led_colors = [(0, 0, 0) for _ in range(self.num_leds)]
        return self.send_data(led_colors, force=True)
    
    def get_debug_messages(self, clear=True):
        """Get and optionally clear debug messages from the Arduino."""
        messages = self.debug_messages.copy()
        if clear:
            self.debug_messages.clear()
        return messages 