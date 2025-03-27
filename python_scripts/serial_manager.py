import serial
import time
import struct
import threading
import logging
import re
import collections

class SerialManager:
    """Manages serial communication with the Arduino controlling the LED strip."""
    
    def __init__(self, port=None, baud_rate=115200, num_leds=60, mock_mode=False, debug_mode=False, max_fps=30):
        self.port = port
        self.baud_rate = baud_rate
        self.num_leds = num_leds
        self.mock_mode = mock_mode
        self.debug_mode = debug_mode
        self.serial_connection = None
        self.auto_reconnect = True
        self.connected = False
        self.lock = threading.Lock()
        self.last_frame_time = 0
        self.frame_count = 0
        self.max_fps = max_fps
        self.target_fps = max_fps  # Dynamic target FPS that adjusts based on performance
        self.min_frame_time = 1.0 / max_fps if max_fps > 0 else 0
        self.debug_messages = collections.deque(maxlen=100)
        self.buffer_check_interval = 0.5  # Check buffer status more frequently
        self.last_buffer_check = 0
        self.buffer_fullness = 0.0  # 0.0 to 1.0
        self.dropped_frames = 0
        self.arduino_fps = 0.0  # FPS reported by Arduino
        
        # Dynamic FPS control parameters
        self.ack_times = []  # Track recent acknowledgment times
        self.max_ack_samples = 10  # Number of samples to keep for averaging
        self.last_adjustment_time = 0
        self.adjustment_interval = 0.5  # How often to adjust FPS (seconds)
        self.fps_step_up = 5  # How much to increase FPS by
        self.fps_step_down = 10  # How much to decrease FPS by
        self.min_target_fps = 10  # Don't go below this FPS
        self.buffer_high_threshold = 0.7  # Buffer fullness threshold to decrease FPS
        self.buffer_low_threshold = 0.3  # Buffer fullness threshold to increase FPS
        self.ack_time_high_threshold = 0.02  # If ack time > 20ms, consider slowing down
        
        # Patterns to extract FPS from Arduino debug messages
        # Will match both 'FPS: 12.3' and 'Frame: 123, FPS: 12.3, Update: 5ms, Dropped: 10'
        self._arduino_fps_regex = re.compile(r'FPS: (\d+\.\d+|\d+)')
        
        # Initialize thread for buffer status checking
        self.buffer_check_thread = None
        self.buffer_check_active = False
        
        # Track acknowledgment times
        self.last_ack_time = 0
        self.current_ack_time = 0
        
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
                
                # Parse FPS information from debug messages
                if 'FPS:' in message:
                    try:
                        # Log the raw message first for debugging
                        logging.info(f"Arduino FPS message: '{message}'")
                        
                        # Try to use regex to extract FPS value for more reliable parsing
                        match = self._arduino_fps_regex.search(message)
                        if match:
                            fps_str = match.group(1)
                            # Check for invalid values
                            if fps_str == '?' or fps_str == 'nan':
                                logging.warning(f"Invalid FPS value in message: '{message}'")
                            else:
                                self.arduino_fps = float(fps_str)
                                logging.info(f"Parsed Arduino FPS: {self.arduino_fps}")
                        else:
                            # Fallback to splitting method if regex fails
                            fps_part = message.split('FPS:')[1].split(',')[0].strip()
                            if fps_part != '?' and fps_part != 'nan':
                                self.arduino_fps = float(fps_part)
                                logging.info(f"Parsed Arduino FPS (fallback): {self.arduino_fps}")
                            else:
                                logging.warning(f"Invalid FPS value in message: '{message}'")
                    except (ValueError, IndexError, AttributeError) as e:
                        logging.warning(f"Failed to parse FPS from message: '{message}' - {str(e)}")
                
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
    
    def adjust_fps(self, ack_time):
        """
        Dynamically adjust the target FPS based on acknowledgment time and buffer fullness.
        
        Args:
            ack_time: Time taken to get acknowledgment from Arduino (seconds)
        """
        # Add the latest acknowledgment time to our samples
        self.ack_times.append(ack_time)
        # Keep only the most recent samples
        self.ack_times = self.ack_times[-self.max_ack_samples:]
        
        # Only adjust periodically to give time for changes to stabilize
        current_time = time.time()
        if current_time - self.last_adjustment_time < self.adjustment_interval:
            return
            
        self.last_adjustment_time = current_time
        
        # Calculate average acknowledgment time
        avg_ack_time = sum(self.ack_times) / len(self.ack_times) if self.ack_times else 0
        
        # Log current status
        logging.debug(f"FPS Control: Current target FPS: {self.target_fps}, Buffer: {self.buffer_fullness * 100:.1f}%, Avg ack time: {avg_ack_time * 1000:.1f}ms")
        
        # Decision logic for FPS adjustment
        if self.buffer_fullness > self.buffer_high_threshold or avg_ack_time > self.ack_time_high_threshold:
            # Buffer is getting full or Arduino is taking too long to respond - slow down
            new_target = max(self.target_fps - self.fps_step_down, self.min_target_fps)
            if new_target < self.target_fps:
                self.target_fps = new_target
                self.min_frame_time = 1.0 / self.target_fps if self.target_fps > 0 else 0
                logging.info(f"FPS Control: Decreasing target FPS to {self.target_fps} (buffer: {self.buffer_fullness * 100:.1f}%, ack: {avg_ack_time * 1000:.1f}ms)")
        elif self.buffer_fullness < self.buffer_low_threshold and avg_ack_time < self.ack_time_high_threshold / 2:
            # Buffer has room and Arduino is responding quickly - speed up
            new_target = min(self.target_fps + self.fps_step_up, self.max_fps)
            if new_target > self.target_fps:
                self.target_fps = new_target
                self.min_frame_time = 1.0 / self.target_fps if self.target_fps > 0 else 0
                logging.info(f"FPS Control: Increasing target FPS to {self.target_fps} (buffer: {self.buffer_fullness * 100:.1f}%, ack: {avg_ack_time * 1000:.1f}ms)")
    
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
        
        # Check buffer status more frequently
        if current_time - self.last_buffer_check > self.buffer_check_interval:
            self.check_buffer_status()
            self.last_buffer_check = current_time
            
            if self.buffer_fullness > 0.9:  # Critical buffer level
                logging.warning(f"Arduino buffer critical at {self.buffer_fullness * 100:.1f}% - dropping frame")
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
                ack_start_time = time.time()
                self.serial_connection.write(data)
                self.serial_connection.flush()
                
                # Process any incoming acknowledgments with timeout
                ack_received = False
                
                while time.time() - ack_start_time < 0.1:  # 100ms timeout
                    self.process_incoming_data()
                    
                    # If we've processed all pending data, we're done
                    if not self.serial_connection.in_waiting:
                        ack_received = True
                        break
                    
                    time.sleep(0.001)  # Small sleep to avoid busy waiting
                
                # Calculate acknowledgment time
                ack_time = time.time() - ack_start_time
                
                # Adjust FPS based on acknowledgment time and buffer status
                if ack_received:
                    self.adjust_fps(ack_time)
                
                self.frame_count += 1
                self.last_frame_time = current_time
                
                if self.frame_count % 100 == 0:
                    actual_fps = 1.0 / (elapsed if elapsed > 0 else 0.001)
                    logging.info(f"Sent frame {self.frame_count}, Actual FPS: {actual_fps:.1f}, Target FPS: {self.target_fps}, Ack time: {ack_time*1000:.1f}ms, Buffer: {self.buffer_fullness*100:.1f}%")
                
                return True
                
            except Exception as e:
                logging.error(f"Error sending data to Arduino: {str(e)}")
                # Decrease target FPS after an error
                self.target_fps = max(self.target_fps - self.fps_step_down, self.min_target_fps)
                self.min_frame_time = 1.0 / self.target_fps if self.target_fps > 0 else 0
                logging.info(f"Error encountered - decreasing target FPS to {self.target_fps}")
                return False
    
    def clear_leds(self):
        """Turn off all LEDs."""
        led_colors = [(0, 0, 0) for _ in range(self.num_leds)]
        return self.send_data(led_colors, force=True)
    
    def get_debug_messages(self, clear=True):
        """Get and optionally clear debug messages from the Arduino."""
        messages = list(self.debug_messages)
        if clear:
            self.debug_messages.clear()
        return messages 