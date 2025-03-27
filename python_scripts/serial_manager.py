import serial
import time
import logging

class ArduinoManager:
    """Manages serial communication with Arduino for LED control"""
    
    def __init__(self, port='/dev/tty.usbserial-110', baud_rate=115200, timeout=1.0):
        """Initialize with serial port settings"""
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_conn = None
        self.is_connected = False
        self.logger = logging.getLogger('ArduinoManager')
    
    def connect(self):
        """Connect to Arduino via serial port"""
        if self.is_connected:
            return True
            
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            
            # Wait for Arduino to reset
            time.sleep(1.0)
            
            # Flush input buffer
            self.serial_conn.flushInput()
            
            self.is_connected = True
            self.logger.info(f"Connected to Arduino on {self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Arduino: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if not self.is_connected:
            return
            
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                
            self.is_connected = False
            self.logger.info("Disconnected from Arduino")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from Arduino: {e}")
    
    def send_data(self, colors):
        """Send LED color data to Arduino
        
        Args:
            colors: List of RGB values as [r1, g1, b1, r2, g2, b2, ...]
        """
        if not self.is_connected or not self.serial_conn:
            self.logger.warning("Cannot send data - not connected to Arduino")
            return False
            
        try:
            # Prepare data format
            # Format: <num_bytes><r1><g1><b1>...<rn><gn><bn>
            num_leds = len(colors) // 3
            
            # Ensure values are integers and in valid range
            valid_colors = [max(0, min(255, int(v))) for v in colors]
            
            data = bytearray(valid_colors)
            header = bytearray([min(num_leds, 255)])
            
            # Send data
            self.serial_conn.write(header + data)
            self.serial_conn.flush()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending data to Arduino: {e}")
            # Try to reconnect
            self.is_connected = False
            return False
    
    def receive_data(self, num_bytes=1, timeout=0.5):
        """Receive data from Arduino"""
        if not self.is_connected or not self.serial_conn:
            return None
            
        try:
            # Set temporary timeout
            original_timeout = self.serial_conn.timeout
            self.serial_conn.timeout = timeout
            
            # Read data
            data = self.serial_conn.read(num_bytes)
            
            # Restore timeout
            self.serial_conn.timeout = original_timeout
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error receiving data from Arduino: {e}")
            return None 