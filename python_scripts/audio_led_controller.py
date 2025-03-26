#!/usr/bin/env python3
"""
Audio LED Controller
Main controller module that integrates audio capture, visualization, and LED control.
"""
import sys
import time
import argparse
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

# Import our modules
try:
    # When running as module (preferred)
    from python_scripts.audio_visualizer import AudioCapture, AudioAnalyzer
    from python_scripts.audio_visualizations import create_visualizer, VISUALIZATIONS
    from python_scripts.animation_sender import LEDAnimationSender
except ImportError:
    # When running as script directly
    from audio_visualizer import AudioCapture, AudioAnalyzer
    from audio_visualizations import create_visualizer, VISUALIZATIONS
    from animation_sender import LEDAnimationSender

# Default parameters
DEFAULT_PARAMETERS = {
    'fps': 30,                  # Target frames per second
    'duration': 0,              # Duration in seconds (0 = infinite)
    'device_name': 'BlackHole', # Audio capture device name
    'vis_type': 'beat_pulse',   # Visualization type
    'brightness': 1.0,          # Overall brightness
    'sensitivity': 1.0,         # Audio sensitivity
    'num_leds': 60,             # Number of LEDs in strip
}


class AudioLEDController:
    """Main controller class for Audio LED Visualization"""
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize controller with parameters
        
        Args:
            params: Dictionary of parameters
        """
        self.params = DEFAULT_PARAMETERS.copy()
        self.params.update(params)
        
        # Components
        self.audio_capture = None
        self.audio_analyzer = None
        self.visualizer = None
        self.led_sender = None
        
        # Runtime variables
        self.is_running = False
        self.frames_processed = 0
        self.start_time = 0
        self.actual_fps = 0
        
    def setup(self) -> bool:
        """Set up all components"""
        try:
            # Create audio capture component
            self.audio_capture = AudioCapture(
                device_name=self.params.get('device_name', 'BlackHole'),
                sample_rate=44100,
                buffer_size=1024,
                channels=2
            )
            
            # Create audio analyzer
            self.audio_analyzer = AudioAnalyzer(
                sample_rate=44100,
                buffer_size=1024
            )
            
            # Create visualizer
            self.visualizer = create_visualizer(
                name=self.params.get('vis_type', 'beat_pulse'),
                num_leds=self.params.get('num_leds', 60)
            )
            
            # Set visualizer parameters
            self.visualizer.set_param('brightness', self.params.get('brightness', 1.0))
            self.visualizer.set_param('sensitivity', self.params.get('sensitivity', 1.0))
            
            if 'color_scheme' in self.params:
                self.visualizer.set_param('color_scheme', self.params.get('color_scheme', 0))
            
            # Create LED sender
            if 'port' in self.params:
                self.led_sender = LEDAnimationSender(
                    port=self.params['port'],
                    baud_rate=115200,
                    num_leds=self.params.get('num_leds', 60)
                )
            else:
                print("Warning: No serial port specified, running in preview mode (no LED output)")
                self.led_sender = None
            
            return True
        
        except Exception as e:
            print(f"Error setting up controller: {e}")
            self.cleanup()
            return False
    
    def start(self) -> bool:
        """Start the audio capture and processing"""
        # Start audio capture
        if not self.audio_capture.start():
            print("Failed to start audio capture")
            return False
        
        # Connect to Arduino if LED sender is available
        if self.led_sender:
            if not self.led_sender.connect():
                print("Failed to connect to Arduino")
                print("Running in preview mode (no LED output)")
                self.led_sender = None
        
        # Set running state
        self.is_running = True
        self.frames_processed = 0
        self.start_time = time.time()
        
        print(f"Started audio LED controller with {self.params['vis_type']} visualization")
        print("Press Ctrl+C to stop")
        
        return True
    
    def run(self) -> None:
        """Main loop for audio processing and visualization"""
        if not self.is_running:
            if not self.start():
                return
        
        try:
            # Calculate frame interval
            target_fps = self.params.get('fps', 30)
            frame_interval = 1.0 / target_fps
            
            # Calculate duration
            duration = self.params.get('duration', 0)
            end_time = self.start_time + duration if duration > 0 else float('inf')
            
            # Main loop
            while self.is_running and time.time() < end_time:
                # Measure frame start time
                frame_start = time.time()
                
                # Process one frame
                self.process_frame()
                
                # Calculate time to sleep to maintain target FPS
                elapsed = time.time() - frame_start
                sleep_time = max(0, frame_interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Update actual FPS
                frame_time = elapsed + sleep_time
                self.actual_fps = 0.9 * self.actual_fps + 0.1 * (1.0 / max(frame_time, 0.001))
                
                # Print status every second
                if self.frames_processed % target_fps == 0:
                    print(f"Frame: {self.frames_processed}, FPS: {self.actual_fps:.1f}")
        
        except KeyboardInterrupt:
            print("\nStopping audio LED controller...")
        
        finally:
            self.cleanup()
    
    def process_frame(self) -> None:
        """Process a single frame of audio data"""
        # Get audio data from capture
        audio_data = self.audio_capture.get_audio_data()
        
        # Analyze audio
        is_beat = self.audio_analyzer.detect_beat(audio_data)
        fft_data = self.audio_analyzer.compute_fft(audio_data)
        bands = self.audio_analyzer.get_frequency_bands(fft_data, num_bands=16)
        
        # Generate visualization
        colors = self.visualizer.update(is_beat, fft_data, bands)
        
        # Send to LEDs if available
        if self.led_sender:
            self.led_sender.send_frame(colors)
        
        # Update stats
        self.frames_processed += 1
    
    def cleanup(self) -> None:
        """Clean up resources"""
        # Stop audio capture
        if self.audio_capture:
            self.audio_capture.stop()
        
        # Disconnect LED sender
        if self.led_sender:
            self.led_sender.disconnect()
        
        # Reset state
        self.is_running = False
        
        # Print final stats
        total_time = time.time() - self.start_time
        if self.frames_processed > 0 and total_time > 0:
            avg_fps = self.frames_processed / total_time
            print(f"\nProcessed {self.frames_processed} frames in {total_time:.1f} seconds")
            print(f"Average FPS: {avg_fps:.1f}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Audio LED Controller')
    
    # Audio capture options
    parser.add_argument('--device', type=str, default='BlackHole',
                        help='Audio capture device name (default: BlackHole)')
    
    # Visualization options
    parser.add_argument('--vis', type=str, choices=list(VISUALIZATIONS.keys()),
                        default='beat_pulse', help='Visualization type')
    parser.add_argument('--brightness', type=float, default=1.0,
                        help='Brightness (0.0-1.0)')
    parser.add_argument('--sensitivity', type=float, default=1.0,
                        help='Audio sensitivity (0.0+)')
    parser.add_argument('--color-scheme', type=int, default=0,
                        help='Color scheme index')
    
    # LED control options
    parser.add_argument('--port', type=str, 
                        help='Serial port for Arduino (required for LED output)')
    parser.add_argument('--leds', type=int, default=60, 
                        help='Number of LEDs in strip')
    
    # Runtime options
    parser.add_argument('--fps', type=int, default=30,
                        help='Target frames per second')
    parser.add_argument('--duration', type=int, default=0,
                        help='Duration in seconds (0 = infinite)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    
    return parser.parse_args()


def list_audio_devices():
    """List all available audio devices"""
    import sounddevice as sd
    
    devices = sd.query_devices()
    print("\nAvailable audio devices:")
    for i, device in enumerate(devices):
        input_ch = device.get('max_input_channels', 0)
        output_ch = device.get('max_output_channels', 0)
        
        if input_ch > 0:
            print(f"{i}: {device['name']} (Input: {input_ch} channels)")
        elif output_ch > 0:
            print(f"{i}: {device['name']} (Output: {output_ch} channels)")
        else:
            print(f"{i}: {device['name']}")
    
    print(f"\nDefault input device: {sd.default.device[0]}")
    print(f"Default output device: {sd.default.device[1]}")


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # List devices if requested
    if args.list_devices:
        list_audio_devices()
        sys.exit(0)
    
    # Extract parameters from arguments
    params = {
        'device_name': args.device,
        'vis_type': args.vis,
        'brightness': args.brightness,
        'sensitivity': args.sensitivity,
        'color_scheme': args.color_scheme,
        'num_leds': args.leds,
        'fps': args.fps,
        'duration': args.duration,
    }
    
    # Add port if specified
    if args.port:
        params['port'] = args.port
    
    # Create controller
    controller = AudioLEDController(params)
    
    # Set up and run
    if controller.setup():
        controller.run()
    else:
        print("Failed to set up controller") 