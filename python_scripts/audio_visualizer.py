#!/usr/bin/env python3
"""
Audio Visualization for LED Strip
Captures system audio and creates visualizations for LED strips.
"""
import time
import numpy as np
import sounddevice as sd
import pyaudio
import librosa
import matplotlib.pyplot as plt
from scipy.fft import fft
from collections import deque
from typing import List, Tuple, Dict, Callable, Optional, Any, Union

# Audio capture constants
SAMPLE_RATE = 44100  # Hz
BUFFER_SIZE = 1024   # Number of audio frames per buffer
CHANNELS = 2         # Stereo
WINDOW_SIZE = 5      # Number of buffers to store in history for smoothing

# LED strip constants (from previous implementation)
NUM_LEDS = 60

class AudioCapture:
    """Handles audio capture from system sources on macOS"""
    
    def __init__(self, device_name: str = "BlackHole", sample_rate: int = SAMPLE_RATE, 
                 buffer_size: int = BUFFER_SIZE, channels: int = CHANNELS):
        """
        Initialize audio capture system.
        
        Args:
            device_name: Name of audio device to capture from (default: "BlackHole")
            sample_rate: Audio sample rate in Hz
            buffer_size: Number of audio frames per buffer
            channels: Number of audio channels (1=mono, 2=stereo)
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.channels = channels
        self.device_name = device_name
        self.device_id = None
        
        # Audio buffer for processing
        self.buffer = np.zeros((self.buffer_size, self.channels))
        
        # Audio history for smoothing
        self.audio_history = deque(maxlen=WINDOW_SIZE)
        
        # Initialize audio capture system
        self.setup_audio_device()
        
        # Stream object
        self.stream = None
        self.py_audio = None
        
    def setup_audio_device(self) -> None:
        """Find and set up the audio device for capture"""
        print(f"Looking for audio device: {self.device_name}")
        
        # List all audio devices
        devices = sd.query_devices()
        print("\nAvailable audio devices:")
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']}")
            
        # Look for BlackHole or specified device
        for i, device in enumerate(devices):
            if self.device_name.lower() in device['name'].lower():
                self.device_id = i
                print(f"\nFound {self.device_name} at device index {i}")
                break
        
        # If not found, use default input device
        if self.device_id is None:
            self.device_id = sd.default.device[0]
            print(f"\nWarning: {self.device_name} not found. Using default device: {devices[self.device_id]['name']}")
            print("To capture system audio, you need to install and configure BlackHole or Soundflower")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio stream"""
        # Convert raw audio data to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Reshape to match channels
        audio_data = audio_data.reshape(-1, self.channels)
        
        # Store in buffer
        self.buffer = audio_data
        
        # Add to history
        self.audio_history.append(audio_data)
        
        # Return empty data and continue flag
        return (None, pyaudio.paContinue)
    
    def start(self) -> bool:
        """Start audio capture"""
        try:
            self.py_audio = pyaudio.PyAudio()
            
            # Open audio stream
            self.stream = self.py_audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_id,
                frames_per_buffer=self.buffer_size,
                stream_callback=self.audio_callback
            )
            
            # Start the stream
            self.stream.start_stream()
            print(f"Audio capture started on device {self.device_id}")
            return True
            
        except Exception as e:
            print(f"Error starting audio capture: {e}")
            self.cleanup()
            return False
    
    def stop(self) -> None:
        """Stop audio capture"""
        self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.stream is not None and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.py_audio is not None:
            self.py_audio.terminate()
            self.py_audio = None
    
    def get_audio_data(self) -> np.ndarray:
        """Get the latest audio buffer"""
        return self.buffer
    
    def get_smoothed_audio_data(self) -> np.ndarray:
        """Get smoothed audio data using history window"""
        if len(self.audio_history) == 0:
            return np.zeros((self.buffer_size, self.channels))
        
        # Calculate average of recent buffers
        return np.mean(self.audio_history, axis=0)


class AudioAnalyzer:
    """Analyzes audio data to extract features for visualization"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, buffer_size: int = BUFFER_SIZE):
        """
        Initialize audio analyzer.
        
        Args:
            sample_rate: Audio sample rate in Hz
            buffer_size: Number of audio frames to analyze at once
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # FFT variables
        self.fft_size = buffer_size
        self.freq_bins = np.fft.rfftfreq(self.fft_size, 1.0/self.sample_rate)
        
        # Beat detection parameters
        self.energy_history = deque(maxlen=50)  # Store energy history for 50 frames
        self.beat_threshold = 1.5   # How much energy increase for a beat
        self.is_beat = False
        self.beat_counted = False
        self.last_beat_time = 0
        self.beat_interval = 0.5    # Minimum time between beats (seconds)
    
    def compute_fft(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Compute FFT on audio data
        
        Args:
            audio_data: Audio data to analyze (n_samples, n_channels)
            
        Returns:
            np.ndarray: Frequency magnitudes
        """
        # Convert to mono if stereo by averaging channels
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            audio_mono = np.mean(audio_data, axis=1)
        else:
            audio_mono = audio_data.flatten()
        
        # Apply window function to reduce spectral leakage
        window = np.hanning(len(audio_mono))
        windowed_data = audio_mono * window
        
        # Compute FFT
        fft_data = np.abs(np.fft.rfft(windowed_data))
        
        # Convert to dB scale (logarithmic amplitude)
        fft_data = 20 * np.log10(fft_data + 1e-10)
        
        return fft_data
    
    def detect_beat(self, audio_data: np.ndarray) -> bool:
        """
        Simple beat detection algorithm
        
        Args:
            audio_data: Audio data to analyze
            
        Returns:
            bool: True if a beat is detected
        """
        # Check for empty audio data
        if audio_data is None or len(audio_data) == 0:
            return False
            
        # Convert to mono and compute energy
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            audio_mono = np.mean(audio_data, axis=1)
        else:
            audio_mono = audio_data.flatten()
        
        # Compute RMS energy
        energy = np.sqrt(np.mean(audio_mono ** 2))
        
        # Add to history
        self.energy_history.append(energy)
        
        # Wait until we have enough history
        if len(self.energy_history) < 20:
            return False
        
        # Get average of recent energies
        recent_energies = list(self.energy_history)[-20:-1]
        if not recent_energies:  # Safeguard against empty list
            return False
            
        avg_energy = np.mean(recent_energies)
        
        # Check if current energy is significantly higher than average
        is_beat_by_energy = energy > avg_energy * self.beat_threshold
        
        # Enforce minimum time between beats
        current_time = time.time()
        enough_time_passed = current_time - self.last_beat_time > self.beat_interval
        
        # Detect beat
        if is_beat_by_energy and enough_time_passed:
            self.last_beat_time = current_time
            self.is_beat = True
            return True
        else:
            self.is_beat = False
            return False

    def get_frequency_bands(self, fft_data: np.ndarray, num_bands: int = 16) -> np.ndarray:
        """
        Divide FFT data into frequency bands
        
        Args:
            fft_data: FFT data
            num_bands: Number of frequency bands to create
            
        Returns:
            np.ndarray: Energy in each frequency band
        """
        # Get number of FFT bins
        n_bins = len(fft_data)
        
        # Create logarithmically spaced bands (more resolution in lower frequencies)
        bands = np.logspace(0, np.log10(max(1, n_bins)), num_bands + 1).astype(int)
        
        # Ensure bands are within range
        bands = np.clip(bands, 0, n_bins - 1)
        
        # Calculate energy in each band
        band_energies = np.zeros(num_bands)
        for i in range(num_bands):
            start = bands[i]
            end = bands[i + 1]
            if start >= end:  # Ensure valid slice
                band_energies[i] = 0
            else:
                # Use np.nanmean to handle NaN values
                slice_data = fft_data[start:end]
                if len(slice_data) > 0:
                    band_energies[i] = np.mean(slice_data)
                else:
                    band_energies[i] = 0
        
        return band_energies


def list_audio_devices():
    """Print a list of all available audio devices"""
    devices = sd.query_devices()
    print("\nAvailable audio devices:")
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']}")
        if device.get('max_input_channels', 0) > 0:
            print(f"  Input channels: {device['max_input_channels']}")
        if device.get('max_output_channels', 0) > 0:
            print(f"  Output channels: {device['max_output_channels']}")
    
    print(f"\nDefault input device: {sd.default.device[0]}")
    print(f"Default output device: {sd.default.device[1]}")
    

if __name__ == "__main__":
    # List all audio devices
    list_audio_devices()
    
    # Test audio capture
    audio_capture = AudioCapture()
    if audio_capture.start():
        print("Audio capture started. Press Ctrl+C to stop...")
        
        # Create analyzer
        analyzer = AudioAnalyzer()
        
        try:
            # Run for 10 seconds
            for _ in range(100):
                # Get audio data
                audio_data = audio_capture.get_audio_data()
                
                # Analyze for beats
                is_beat = analyzer.detect_beat(audio_data)
                
                # Compute FFT
                fft_data = analyzer.compute_fft(audio_data)
                
                # Get frequency bands
                bands = analyzer.get_frequency_bands(fft_data, num_bands=8)
                
                # Print status
                if is_beat:
                    print("BEAT! Energy bands:", bands)
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Stopping audio capture...")
        finally:
            audio_capture.stop()
    else:
        print("Failed to start audio capture") 