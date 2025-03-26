#!/usr/bin/env python3
"""
Audio Visualization Test
Test script that creates synthetic audio data to test visualizations
without requiring an actual audio device.
"""
import sys
import time
import numpy as np
import argparse
from typing import List, Tuple

try:
    from python_scripts.audio_visualizer import AudioAnalyzer
    from python_scripts.audio_visualizations import create_visualizer, VISUALIZATIONS
    from python_scripts.animation_sender import LEDAnimationSender
except ImportError:
    from audio_visualizer import AudioAnalyzer
    from audio_visualizations import create_visualizer, VISUALIZATIONS
    from animation_sender import LEDAnimationSender


class MockAudioGenerator:
    """Generate synthetic audio data for testing visualizations"""
    
    def __init__(self, sample_rate=44100, buffer_size=1024):
        """Initialize the mock audio generator"""
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.time_index = 0
        self.beat_interval = 0.5  # beat every 0.5 seconds
        self.last_beat_time = 0
        
    def generate_sine_wave(self, frequency, amplitude=0.5):
        """Generate a sine wave of the given frequency"""
        t = np.arange(self.time_index, 
                      self.time_index + self.buffer_size) / self.sample_rate
        return amplitude * np.sin(2 * np.pi * frequency * t)
    
    def generate_audio_data(self):
        """Generate synthetic audio data"""
        # Base frequencies for testing
        bass_freq = 80
        mid_freq = 440
        high_freq = 2000
        
        # Time for this buffer
        current_time = self.time_index / self.sample_rate
        
        # Generate individual components
        bass = self.generate_sine_wave(bass_freq, 0.7)
        mid = self.generate_sine_wave(mid_freq, 0.5)
        high = self.generate_sine_wave(high_freq, 0.3)
        
        # Add some noise
        noise = np.random.normal(0, 0.05, self.buffer_size)
        
        # Create beat effect
        beat_amplitude = 0
        if current_time - self.last_beat_time >= self.beat_interval:
            beat_amplitude = 0.8
            self.last_beat_time = current_time
        
        beat = beat_amplitude * np.exp(-5 * np.linspace(0, 1, self.buffer_size))
        
        # Mix all components
        audio_data = bass + mid + high + noise + beat
        
        # Normalize to range [-1, 1]
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Update time index
        self.time_index += self.buffer_size
        
        return audio_data


def run_test(vis_type, num_leds, port=None, duration=10):
    """Run the audio visualization test"""
    # Create components
    mock_audio = MockAudioGenerator()
    analyzer = AudioAnalyzer()
    visualizer = create_visualizer(vis_type, num_leds)
    
    # Create LED sender if port is provided
    led_sender = None
    if port:
        led_sender = LEDAnimationSender(port=port, num_leds=num_leds)
        if not led_sender.connect():
            print(f"Failed to connect to port {port}")
            led_sender = None
            
    # Set up test parameters
    start_time = time.time()
    end_time = start_time + duration
    frame_count = 0
    
    print(f"Running audio test with {vis_type} visualization for {duration} seconds")
    print("Press Ctrl+C to stop")
    
    try:
        # Main test loop
        while time.time() < end_time:
            loop_start = time.time()
            
            # Generate mock audio data
            audio_data = mock_audio.generate_audio_data()
            
            # Analyze audio
            is_beat = analyzer.detect_beat(audio_data)
            fft_data = analyzer.compute_fft(audio_data)
            bands = analyzer.get_frequency_bands(fft_data)
            
            # Update visualization
            colors = visualizer.update(is_beat, fft_data, bands)
            
            # Send to LEDs if available
            if led_sender:
                led_sender.send_frame(colors)
            
            # Print status
            if is_beat:
                print("BEAT", end="", flush=True)
            else:
                print(".", end="", flush=True)
            
            if frame_count % 30 == 0:
                print(f" Frame {frame_count}")
            
            # Maintain frame rate (30 fps)
            elapsed = time.time() - loop_start
            if elapsed < 1/30:
                time.sleep(1/30 - elapsed)
                
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    
    finally:
        # Clean up
        if led_sender:
            led_sender.disconnect()
            
        # Print results
        total_time = time.time() - start_time
        print(f"\nProcessed {frame_count} frames in {total_time:.1f} seconds")
        print(f"Average FPS: {frame_count / total_time:.1f}")
        
        return True


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Audio Visualization Test')
    
    parser.add_argument('--vis', type=str, choices=list(VISUALIZATIONS.keys()),
                       default='beat_pulse', help='Visualization type to test')
    parser.add_argument('--leds', type=int, default=60,
                       help='Number of LEDs')
    parser.add_argument('--port', type=str,
                       help='Serial port for Arduino (optional)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Test duration in seconds')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_test(args.vis, args.leds, args.port, args.duration) 