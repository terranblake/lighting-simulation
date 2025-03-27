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
    
    def __init__(self, device_name="BlackHole", device_id=None, channels=1, sample_rate=48000, buffer_size=1024):
        """Initialize audio capture with the specified settings"""
        self.device_name = device_name
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size
        self.stream = None
        self.is_running = False
        self.device_info = None
        self.is_input = True  # Default to input device
        
        # Use a deque for thread-safe audio buffer
        self.audio_buffer = deque(maxlen=5)
        
    def setup_audio_device(self):
        """Find and set up the audio device"""
        try:
            # List available devices
            devices = sd.query_devices()
            
            # Look for BlackHole or the specified device
            device_id = None
            for i, device in enumerate(devices):
                if self.device_name.lower() in device['name'].lower():
                    device_id = i
                    print(f"Found {self.device_name} at device index {device_id}")
                    break
            
            # If device not found, use default input
            if device_id is None:
                print(f"Warning: {self.device_name} not found. Using default input device.")
                device_id = sd.default.device[0]
                # Get the name of the default device
                self.device_name = devices[device_id]['name']
                print(f"Using {self.device_name} (index {device_id})")
            
            # Set the device
            self.device_info = devices[device_id]
            
            # Determine channels - handle different device types
            if self.device_info['max_input_channels'] > 0:
                # It's an input device, use up to requested channels
                self.channels = min(self.channels, self.device_info['max_input_channels'])
                self.device_id = device_id
                self.is_input = True
            elif self.device_info['max_output_channels'] > 0:
                # It's an output device (like BlackHole), use monitor mode
                self.channels = min(self.channels, self.device_info['max_output_channels'])
                self.device_id = device_id
                self.is_input = False
                print(f"Note: {self.device_name} is an output device. Will attempt to capture in monitor mode.")
            else:
                raise ValueError(f"Device {self.device_name} has no input or output channels")
            
            return True
            
        except Exception as e:
            print(f"Error setting up audio device: {e}")
            return False
            
    def start(self):
        """Start the audio capture stream"""
        if not self.setup_audio_device():
            return False
            
        try:
            # Configure the stream based on device type
            if self.is_input:
                # Input device configuration
                self.stream = sd.InputStream(
                    device=self.device_id,
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    blocksize=self.buffer_size,
                    callback=self._audio_callback_input
                )
            else:
                # Output device configuration (monitor mode)
                self.stream = sd.Stream(
                    device=(sd.default.device[0], self.device_id),  # Input and output devices
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    blocksize=self.buffer_size,
                    callback=self._audio_callback_stream
                )
            
            self.stream.start()
            self.is_running = True
            print(f"Started audio capture from {self.device_name} ({self.channels} channels, {self.sample_rate} Hz)")
            return True
            
        except Exception as e:
            print(f"Error starting audio capture: {e}")
            if "Invalid number of channels" in str(e):
                print(f"The device {self.device_name} doesn't support {self.channels} channels.")
                print(f"Try setting channels=1 or check device capabilities.")
            
            if "BlackHole" in self.device_name and not self.is_input:
                print("To capture system audio with BlackHole:")
                print("1. Open System Settings > Sound")
                print("2. Create a Multi-Output Device including both your speakers and BlackHole")
                print("3. Set the Multi-Output Device as your system output")
            
            return False
    
    def stop(self) -> None:
        """Stop audio capture"""
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.is_running = False
        except Exception as e:
            print(f"Error stopping audio capture: {e}")
    
    def _audio_callback_input(self, indata, frames, time, status):
        """Callback function for input stream"""
        try:
            if status:
                print(f"Audio callback status: {status}")
            
            # Store the audio data
            self.audio_buffer.append(indata.copy())
                    
            # Keep buffer at a reasonable size
            while len(self.audio_buffer) > 5:
                self.audio_buffer.pop(0)
                
        except Exception as e:
            print(f"Error in audio callback: {e}")
    
    def _audio_callback_stream(self, indata, outdata, frames, time, status):
        """Callback function for duplex stream"""
        try:
            if status:
                print(f"Audio callback status: {status}")
            
            # For output monitoring, we'll get audio from outdata if available
            # But we also need to keep indata as a fallback
            if outdata is not None and not np.all(outdata == 0):
                self.audio_buffer.append(outdata.copy())
            else:
                self.audio_buffer.append(indata.copy())
                    
            # Keep buffer at a reasonable size
            while len(self.audio_buffer) > 5:
                self.audio_buffer.pop(0)
                
        except Exception as e:
            print(f"Error in audio callback: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.stream is not None and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.py_audio is not None:
            self.py_audio.terminate()
            self.py_audio = None
    
    def get_audio_data(self):
        """Get the latest audio data for processing"""
        if not self.is_running or len(self.audio_buffer) == 0:
            return None
            
        # Get the most recent audio data
        audio_data = self.audio_buffer[-1]
        
        # Convert to numpy array if needed and ensure shape is correct
        if isinstance(audio_data, np.ndarray):
            # Make sure we're returning mono data if requested
            if audio_data.ndim > 1 and audio_data.shape[1] > 1 and self.channels == 1:
                # Average the channels to get mono
                return np.mean(audio_data, axis=1)
            return audio_data.flatten() if audio_data.ndim > 1 and self.channels == 1 else audio_data
        
        return None
    
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