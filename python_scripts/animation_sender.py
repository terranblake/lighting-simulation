#!/usr/bin/env python3
import argparse
import time
import serial
import math
import sys
from typing import List, Tuple, Callable

# Protocol definitions
HEADER_MARKER = 0xAA
ACK_SUCCESS = 0x01
ACK_ERROR = 0xFF
DEBUG_PRINT = 0x02

# Animation functions
def rainbow_animation(num_leds: int, step: int) -> List[Tuple[int, int, int]]:
    """Generate a rainbow pattern that moves with each step"""
    colors = []
    for i in range(num_leds):
        # Calculate hue value (0-255) based on LED position and animation step
        hue = (i * 256 // num_leds + step) % 256
        # Convert HSV to RGB (simplified conversion)
        if hue < 85:
            r, g, b = 255 - hue * 3, hue * 3, 0
        elif hue < 170:
            hue -= 85
            r, g, b = 0, 255 - hue * 3, hue * 3
        else:
            hue -= 170
            r, g, b = hue * 3, 0, 255 - hue * 3
        colors.append((r, g, b))
    return colors

def color_wipe_animation(num_leds: int, step: int) -> List[Tuple[int, int, int]]:
    """Create a color wipe animation that moves along the strip"""
    colors = [(0, 0, 0)] * num_leds  # Start with all LEDs off
    color_options = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
    ]
    
    # Determine current color from step
    color_index = (step // num_leds) % len(color_options)
    active_color = color_options[color_index]
    
    # Determine how many LEDs to light
    position = step % (num_leds * 2)
    if position >= num_leds:
        # Going backwards
        lit_position = 2 * num_leds - position - 1
    else:
        # Going forwards
        lit_position = position
    
    # Set the color for the active LED
    colors[lit_position] = active_color
    
    return colors

def pulse_animation(num_leds: int, step: int) -> List[Tuple[int, int, int]]:
    """Create a pulsing animation where brightness varies over time"""
    # Calculate brightness as a sine wave for smooth transitions
    brightness = int(128 + 127 * math.sin(step / 10))
    
    # Calculate hue from step (slowly changing color)
    hue = (step // 3) % 256
    
    # Convert HSV to RGB (simplified)
    if hue < 85:
        r, g, b = 255 - hue * 3, hue * 3, 0
    elif hue < 170:
        hue -= 85
        r, g, b = 0, 255 - hue * 3, hue * 3
    else:
        hue -= 170
        r, g, b = hue * 3, 0, 255 - hue * 3
    
    # Scale by brightness
    r = r * brightness // 255
    g = g * brightness // 255
    b = b * brightness // 255
    
    # Return the same color for all LEDs
    return [(r, g, b)] * num_leds

# Available animations
ANIMATIONS = {
    'rainbow': rainbow_animation,
    'color_wipe': color_wipe_animation,
    'pulse': pulse_animation,
}

class LEDAnimationSender:
    def __init__(self, port: str, baud_rate: int = 115200, num_leds: int = 60, debug: bool = True):
        self.port = port
        self.baud_rate = baud_rate
        self.num_leds = num_leds
        self.serial = None
        self.debug = debug
    
    def connect(self) -> bool:
        """Connect to the Arduino over serial"""
        try:
            print(f"Connecting to {self.port} at {self.baud_rate} baud...")
            self.serial = serial.Serial(self.port, self.baud_rate, timeout=5)
            
            # Flush any existing data
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            # Wait for Arduino to reset
            print("Waiting for Arduino to reset (2 seconds)...")
            time.sleep(2)
            
            # Wait for ready signal
            ready = False
            start_time = time.time()
            print("Waiting for READY signal from Arduino...")
            
            while not ready and time.time() - start_time < 5:
                if self.serial.in_waiting > 0:
                    # Read the first byte to check if it's a debug message
                    cmd = self.serial.read(1)
                    if cmd and cmd[0] == DEBUG_PRINT:
                        # This is a debug message, read and display it
                        line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                        print(f"Arduino: {line}")
                        if line == "READY":
                            ready = True
                    else:
                        # Not a debug message, just print the byte
                        print(f"Received unknown byte: {cmd.hex()}")
                time.sleep(0.1)
            
            if ready:
                print("Arduino is ready!")
            else:
                print("Timed out waiting for Arduino READY signal")
                
            return ready
        except Exception as e:
            print(f"Error connecting to Arduino: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial and self.serial.is_open:
            print("Closing serial connection...")
            self.serial.close()
    
    def process_incoming_data(self):
        """Process any incoming data from the serial port"""
        while self.serial and self.serial.in_waiting > 0:
            cmd = self.serial.read(1)
            if not cmd:
                break
                
            if cmd[0] == DEBUG_PRINT:
                # This is a debug message, read and display it
                line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                print(f"Arduino: {line}")
            else:
                # Not a debug message, return the byte
                return cmd[0]
        
        return None
    
    def send_frame(self, colors: List[Tuple[int, int, int]]) -> bool:
        """Send a frame of LED colors to the Arduino"""
        if not self.serial or not self.serial.is_open:
            print("Serial connection not open")
            return False
        
        # First, process any pending incoming data
        while self.process_incoming_data() is not None:
            pass
        
        # Number of LEDs to update (limited to max 60)
        num_to_update = min(len(colors), self.num_leds)
        
        # Create data packet
        data = bytearray([HEADER_MARKER, num_to_update])
        
        # Add RGB values for each LED
        for r, g, b in colors[:num_to_update]:
            data.extend([r, g, b])
        
        # Send data
        try:
            bytes_written = self.serial.write(data)
            self.serial.flush()
            
            if self.debug:
                print(f"Sent {bytes_written} bytes: Header=0x{HEADER_MARKER:02X}, LEDs={num_to_update}")
            
            # Wait for acknowledgment
            start_time = time.time()
            while time.time() - start_time < 2:
                response = self.process_incoming_data()
                
                if response is not None:
                    if response == ACK_SUCCESS:
                        return True
                    elif response == ACK_ERROR:
                        if self.debug:
                            print("Received ERROR response from Arduino")
                        return False
                    else:
                        if self.debug:
                            print(f"Received unexpected response: 0x{response:02X}")
                
                time.sleep(0.01)
            
            if self.debug:
                print("Timeout waiting for acknowledgment")
            return False  # Timeout waiting for acknowledgment
        except Exception as e:
            print(f"Error sending frame: {e}")
            return False

    def run_animation(self, animation_func: Callable, duration: int = 10, fps: int = 30):
        """Run an animation for a specified duration at the given frame rate"""
        if not self.connect():
            print("Failed to connect to Arduino")
            return
        
        try:
            frames = int(duration * fps)
            frame_delay = 1.0 / fps
            
            print(f"Running animation for {duration} seconds at {fps} FPS")
            
            success_count = 0
            fail_count = 0
            
            for step in range(frames):
                start_time = time.time()
                
                # Generate animation frame
                colors = animation_func(self.num_leds, step)
                
                # Send frame to Arduino
                success = self.send_frame(colors)
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    print(f"Warning: Failed to send frame {step}")
                
                # Calculate sleep time to maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                if step % fps == 0:  # Print status every second
                    actual_fps = 1.0 / (elapsed + sleep_time) if (elapsed + sleep_time) > 0 else 0
                    print(f"Step {step}, Actual FPS: {actual_fps:.2f}")
            
            print(f"Animation complete. Successful frames: {success_count}, Failed frames: {fail_count}")
            
        finally:
            self.disconnect()

def main():
    parser = argparse.ArgumentParser(description='LED Animation Controller')
    parser.add_argument('--port', required=True, help='Serial port for Arduino')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--leds', type=int, default=60, help='Number of LEDs')
    parser.add_argument('--animation', choices=ANIMATIONS.keys(), default='rainbow', 
                        help='Animation pattern to display')
    parser.add_argument('--duration', type=int, default=10, help='Animation duration in seconds')
    parser.add_argument('--fps', type=int, default=30, help='Target frames per second')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Create animation sender
    sender = LEDAnimationSender(args.port, args.baud, args.leds, args.debug)
    
    # Run selected animation
    animation_func = ANIMATIONS[args.animation]
    sender.run_animation(animation_func, args.duration, args.fps)

if __name__ == "__main__":
    main() 