#!/usr/bin/env python3
"""
Audio Test Module
Provides synthetic audio data for testing visualizations without a real audio source
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
    """Generates synthetic audio data for testing audio visualizations"""
    
    def __init__(self, sample_rate=48000, buffer_size=1024, channels=1):
        """Initialize with audio settings"""
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.channels = channels
        
        # Settings for generated audio
        self.base_freq = 440.0  # Base frequency in Hz
        self.beat_interval = 0.5  # Beat every 0.5 seconds
        self.freq_shift = 0.0  # Current frequency shift
        self.last_beat_time = 0.0
        
        # Frequency components for generated waveform
        self.freq_components = [
            (1.0, 0.7),     # Base frequency, amplitude
            (2.0, 0.3),     # 1st harmonic
            (3.0, 0.15),    # 2nd harmonic
            (0.5, 0.2),     # Subharmonic
        ]
        
        # Time tracking
        self.time_offset = 0.0
        
    def generate_audio_data(self):
        """Generate synthetic audio data with various frequency components"""
        # Create time array
        t = np.linspace(
            self.time_offset, 
            self.time_offset + self.buffer_size / self.sample_rate, 
            self.buffer_size
        )
        
        # Update time offset for next call
        self.time_offset += self.buffer_size / self.sample_rate
        
        # Check if it's time for a beat
        current_time = time.time()
        is_beat = (current_time - self.last_beat_time) >= self.beat_interval
        if is_beat:
            self.last_beat_time = current_time
            # Shift frequency on beat
            self.freq_shift = (self.freq_shift + 50) % 200 - 100
        
        # Generate waveform with multiple frequency components
        audio = np.zeros(self.buffer_size)
        
        # Add sine waves of different frequencies
        for freq_multiple, amplitude in self.freq_components:
            freq = self.base_freq * freq_multiple + self.freq_shift
            audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add some noise
        noise_level = 0.05
        audio += noise_level * np.random.randn(self.buffer_size)
        
        # Ensure audio is in range [-1, 1]
        audio = np.clip(audio / max(1.0, np.max(np.abs(audio))), -1.0, 1.0)
        
        # When a beat happens, add an attack transient
        if is_beat:
            beat_env = np.exp(-np.linspace(0, 10, self.buffer_size))
            audio += 0.3 * beat_env * np.random.randn(self.buffer_size)
            # Reclip to ensure we're still in range
            audio = np.clip(audio, -1.0, 1.0)
        
        # Reshape for multiple channels if needed
        if self.channels > 1:
            audio = np.tile(audio.reshape(-1, 1), (1, self.channels))
        
        return audio
    
    def get_audio_data(self):
        """API compatible with AudioCapture"""
        return self.generate_audio_data()
        
    def start(self):
        """API compatible with AudioCapture"""
        return True
    
    def stop(self):
        """API compatible with AudioCapture"""
        pass


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
    # Test the mock audio generator
    import matplotlib.pyplot as plt
    
    generator = MockAudioGenerator()
    
    # Generate 10 buffers of audio
    all_audio = []
    for i in range(10):
        audio = generator.generate_audio_data()
        all_audio.append(audio)
        
    # Concatenate and plot
    audio_concat = np.concatenate(all_audio)
    
    plt.figure(figsize=(10, 6))
    plt.plot(audio_concat)
    plt.title("Mock Audio Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.savefig("mock_audio.png")
    plt.close()
    
    print("Generated mock audio and saved waveform plot to mock_audio.png")
    
    args = parse_args()
    run_test(args.vis, args.leds, args.port, args.duration) 