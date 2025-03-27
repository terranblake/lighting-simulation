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
    """
    Class to analyze audio data, compute FFT and beat detection
    """
    def __init__(self, sample_rate=44100, fft_size=1024):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.window = np.hanning(fft_size)
        self.energy_history = []
        self.max_history_size = 43  # ~1 second at 44.1kHz with hop of 1024
        self.beat_threshold = 1.5   # Energy must be this many times the average to be a beat
        self.smoothing = 0.85       # Smoothing factor for energy history
        
        # Precompute frequency bins for efficiency
        self.frequencies = np.fft.rfftfreq(fft_size, 1.0/sample_rate)
        self.frequency_bands_indices = []
        
        # Define frequency band ranges (in Hz)
        self.band_ranges = [
            (20, 60),     # Sub bass
            (60, 250),    # Bass
            (250, 500),   # Low midrange
            (500, 2000),  # Midrange
            (2000, 4000), # Upper midrange
            (4000, 6000), # Presence
            (6000, 20000) # Brilliance
        ]
        
        # Precompute frequency band indices
        for low, high in self.band_ranges:
            indices = np.where((self.frequencies >= low) & (self.frequencies <= high))[0]
            self.frequency_bands_indices.append(indices)
    
    def compute_fft(self, audio_data):
        """
        Compute the FFT of the given audio data
        Returns magnitude spectrum
        """
        # Ensure audio data is the right shape
        if len(audio_data) < self.fft_size:
            # Pad with zeros if needed
            audio_data = np.pad(audio_data, (0, self.fft_size - len(audio_data)))
        elif len(audio_data) > self.fft_size:
            # Truncate if needed
            audio_data = audio_data[:self.fft_size]
        
        # Apply window to reduce spectral leakage
        windowed_data = audio_data * self.window
        
        # Compute FFT - use rfft for real-valued signals (faster)
        fft = np.fft.rfft(windowed_data)
        
        # Get magnitude spectrum (absolute value of complex FFT)
        # Note: computing magnitude once is more efficient
        magnitude = np.abs(fft) / (self.fft_size / 2)
        
        return magnitude
    
    def get_frequency_bands(self, fft_data):
        """
        Divide the FFT data into frequency bands
        Returns list of energies for each band
        """
        # Use precomputed indices for faster band calculation
        bands = [np.mean(fft_data[indices]) if len(indices) > 0 else 0 
                 for indices in self.frequency_bands_indices]
        
        # Apply some scaling for visualization
        bands = np.array(bands) * 2.0
        
        return bands
    
    def detect_beat(self, audio_data):
        """
        Detect beats in the audio data using energy-based approach
        Returns True if beat was detected, False otherwise
        """
        # Calculate RMS energy of the frame
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)  # Convert stereo to mono
            
        # Use vectorized energy calculation (faster than loop)
        energy = np.sqrt(np.mean(audio_data ** 2))
        
        # Add to history
        self.energy_history.append(energy)
        
        # Trim history if needed
        if len(self.energy_history) > self.max_history_size:
            self.energy_history = self.energy_history[-self.max_history_size:]
        
        # Need some history to detect beats
        if len(self.energy_history) < 2:
            return False
        
        # Calculate the average energy over history
        avg_energy = np.mean(self.energy_history)
        
        # Beat is detected if current energy is significantly above average
        is_beat = energy > avg_energy * self.beat_threshold and energy > 0.01
        
        return is_beat


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
                bands = analyzer.get_frequency_bands(fft_data)
                
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