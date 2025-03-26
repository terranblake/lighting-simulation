#!/usr/bin/env python3
"""
Audio Visualizer Demo
A demonstration of the audio visualization system using matplotlib
"""
import sys
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import our modules
try:
    # When running as module (preferred)
    from python_scripts.audio_visualizer import AudioCapture, AudioAnalyzer, list_audio_devices
    from python_scripts.audio_visualizations import create_visualizer, VISUALIZATIONS
except ImportError:
    # When running as script directly
    from audio_visualizer import AudioCapture, AudioAnalyzer, list_audio_devices
    from audio_visualizations import create_visualizer, VISUALIZATIONS

# Default parameters
DEFAULT_VIS_TYPE = 'beat_pulse'
DEFAULT_DEVICE = 'BlackHole'
DEFAULT_NUM_LEDS = 60
DEFAULT_FPS = 30
DEFAULT_DURATION = 0  # 0 means run until manually stopped

class AudioVisualizerDemo:
    """Demo class for audio visualization using matplotlib"""
    
    def __init__(self, params):
        """Initialize the demo with parameters"""
        self.params = params
        self.num_leds = params.get('num_leds', DEFAULT_NUM_LEDS)
        self.device_name = params.get('device_name', DEFAULT_DEVICE)
        self.vis_type = params.get('vis_type', DEFAULT_VIS_TYPE)
        
        # Components
        self.audio_capture = None
        self.audio_analyzer = None
        self.visualizer = None
        
        # Matplotlib objects
        self.fig = None
        self.ax_leds = None
        self.ax_spectrum = None
        self.ax_waveform = None
        self.led_display = None
        self.spectrum_bars = None
        self.waveform_line = None
        self.beat_indicator = None
        
        # Runtime state
        self.is_running = False
        self.frames_processed = 0
        self.start_time = 0
        self.latest_audio_data = None
        self.latest_fft = None
        self.latest_bands = None
        self.latest_beat = False
        self.latest_colors = None
    
    def setup(self):
        """Set up audio capture, analyzer and visualizer"""
        try:
            # Create audio capture component
            self.audio_capture = AudioCapture(
                device_name=self.device_name,
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
                name=self.vis_type,
                num_leds=self.num_leds
            )
            
            # Set visualizer parameters
            self.visualizer.set_param('brightness', self.params.get('brightness', 1.0))
            self.visualizer.set_param('sensitivity', self.params.get('sensitivity', 1.0))
            
            if 'color_scheme' in self.params:
                self.visualizer.set_param('color_scheme', self.params.get('color_scheme', 0))
            
            # Initialize matplotlib figure
            self.setup_plot()
            
            return True
            
        except Exception as e:
            print(f"Error setting up demo: {e}")
            return False
    
    def setup_plot(self):
        """Set up matplotlib figure and axes"""
        # Create figure with multiple subplots
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.suptitle(f"Audio Visualizer Demo - {self.vis_type.replace('_', ' ').title()}")
        
        # LED strip visualization (top subplot)
        self.ax_leds = plt.subplot2grid((3, 1), (0, 0), rowspan=1)
        self.ax_leds.set_title("LED Strip Visualization")
        self.ax_leds.set_xlim(0, self.num_leds - 1)
        self.ax_leds.set_ylim(0, 1)
        self.ax_leds.set_yticks([])
        
        # Create initial LED display (scatter plot with colors)
        x = np.arange(self.num_leds)
        y = np.ones(self.num_leds) * 0.5
        colors = [(0, 0, 0)] * self.num_leds
        self.led_display = self.ax_leds.scatter(x, y, c=colors, s=100, marker='s')
        
        # Beat indicator
        self.beat_indicator = self.ax_leds.text(
            self.num_leds * 0.5, 0.5, "BEAT", 
            fontsize=20, ha='center', va='center', alpha=0,
            bbox=dict(boxstyle="round", fc="yellow", ec="orange", alpha=0.5)
        )
        
        # Frequency spectrum visualization (middle subplot)
        self.ax_spectrum = plt.subplot2grid((3, 1), (1, 0), rowspan=1)
        self.ax_spectrum.set_title("Frequency Spectrum")
        self.ax_spectrum.set_xlim(0, 15)  # 16 frequency bands
        self.ax_spectrum.set_ylim(0, 1)
        self.ax_spectrum.set_xlabel("Frequency Band")
        self.ax_spectrum.set_ylabel("Magnitude")
        
        # Create initial spectrum bars
        x = np.arange(16)
        y = np.zeros(16)
        self.spectrum_bars = self.ax_spectrum.bar(x, y, width=0.8)
        
        # Waveform visualization (bottom subplot)
        self.ax_waveform = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
        self.ax_waveform.set_title("Audio Waveform")
        self.ax_waveform.set_xlim(0, 1023)
        self.ax_waveform.set_ylim(-1, 1)
        self.ax_waveform.set_xlabel("Sample")
        self.ax_waveform.set_ylabel("Amplitude")
        
        # Create initial waveform line
        x = np.arange(1024)
        y = np.zeros(1024)
        self.waveform_line, = self.ax_waveform.plot(x, y, '-', lw=1)
        
        # Adjust layout
        plt.tight_layout()
    
    def start(self):
        """Start audio capture and processing"""
        if not self.audio_capture.start():
            print("Failed to start audio capture")
            return False
        
        self.is_running = True
        self.frames_processed = 0
        self.start_time = time.time()
        
        print(f"Started audio visualizer demo with {self.vis_type} visualization")
        print("Close the plot window to stop")
        
        return True
    
    def update_plot(self, frame):
        """Update the plot with new audio data (called by matplotlib animation)"""
        # Get audio data
        self.latest_audio_data = self.audio_capture.get_audio_data()
        
        if self.latest_audio_data is None or len(self.latest_audio_data) == 0:
            return (self.led_display, *self.spectrum_bars, self.waveform_line, self.beat_indicator)
        
        # Analyze audio
        self.latest_beat = self.audio_analyzer.detect_beat(self.latest_audio_data)
        self.latest_fft = self.audio_analyzer.compute_fft(self.latest_audio_data)
        self.latest_bands = self.audio_analyzer.get_frequency_bands(self.latest_fft, num_bands=16)
        
        # Update visualizer and get colors
        self.latest_colors = self.visualizer.update(
            self.latest_beat, 
            self.latest_fft, 
            self.latest_bands
        )
        
        # Convert colors from 0-255 range to 0-1 range for matplotlib
        normalized_colors = [(r/255, g/255, b/255) for r, g, b in self.latest_colors]
        
        # Update LED display
        self.led_display.set_color(normalized_colors)
        
        # Update spectrum bars
        for i, bar in enumerate(self.spectrum_bars):
            if i < len(self.latest_bands):
                bar.set_height(self.latest_bands[i])
                # Color the bars to match the LEDs where possible
                if i < len(normalized_colors):
                    bar.set_color(normalized_colors[i])
        
        # Update waveform
        self.waveform_line.set_ydata(self.latest_audio_data[:1024])
        
        # Update beat indicator
        if self.latest_beat:
            self.beat_indicator.set_alpha(1.0)
        else:
            # Fade out
            current_alpha = self.beat_indicator.get_alpha()
            self.beat_indicator.set_alpha(max(0, current_alpha - 0.1))
        
        # Update frame counter
        self.frames_processed += 1
        
        # Calculate and display FPS every 30 frames
        if self.frames_processed % 30 == 0:
            elapsed = time.time() - self.start_time
            fps = self.frames_processed / elapsed
            self.fig.suptitle(
                f"Audio Visualizer Demo - {self.vis_type.replace('_', ' ').title()} - "
                f"FPS: {fps:.1f}"
            )
        
        # Return all artists that were modified
        return (self.led_display, *self.spectrum_bars, self.waveform_line, self.beat_indicator)
    
    def run(self):
        """Run the demo animation"""
        if not self.is_running:
            if not self.start():
                return
        
        # Create animation
        animation = FuncAnimation(
            self.fig, 
            self.update_plot, 
            interval=1000//DEFAULT_FPS,  # in milliseconds
            blit=True
        )
        
        # Show plot (blocks until window is closed)
        plt.show()
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.audio_capture:
            self.audio_capture.stop()
        
        self.is_running = False
        
        # Print final stats
        total_time = time.time() - self.start_time
        if self.frames_processed > 0 and total_time > 0:
            avg_fps = self.frames_processed / total_time
            print(f"\nProcessed {self.frames_processed} frames in {total_time:.1f} seconds")
            print(f"Average FPS: {avg_fps:.1f}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Audio Visualizer Demo')
    
    # Audio capture options
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE,
                        help=f'Audio capture device name (default: {DEFAULT_DEVICE})')
    
    # Visualization options
    parser.add_argument('--vis', type=str, choices=list(VISUALIZATIONS.keys()),
                        default=DEFAULT_VIS_TYPE, help='Visualization type')
    parser.add_argument('--brightness', type=float, default=1.0,
                        help='Brightness (0.0-1.0)')
    parser.add_argument('--sensitivity', type=float, default=1.0,
                        help='Audio sensitivity (0.0+)')
    parser.add_argument('--color-scheme', type=int, default=0,
                        help='Color scheme index')
    parser.add_argument('--leds', type=int, default=DEFAULT_NUM_LEDS, 
                        help=f'Number of LEDs to simulate (default: {DEFAULT_NUM_LEDS})')
    
    # Utility options
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices and exit')
    
    return parser.parse_args()


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
    }
    
    # Create and run demo
    demo = AudioVisualizerDemo(params)
    if demo.setup():
        demo.run()
    else:
        print("Failed to set up demo") 