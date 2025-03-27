#!/usr/bin/env python3
"""
Audio Visualizations
Provides visualizations for LED strips based on audio input
"""
import time
import math
import numpy as np
from typing import List, Dict, Tuple, Union, Any, Optional

# LED strip constants
NUM_LEDS = 60

# Color constants
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)
CYAN = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class BaseVisualizer:
    """Base class for audio visualizers"""
    
    def __init__(self, num_leds: int = NUM_LEDS):
        """Initialize with number of LEDs"""
        self.num_leds = num_leds
        self.led_colors = [(0, 0, 0)] * num_leds
        self.params = {
            'brightness': 1.0,
            'sensitivity': 1.0,
            'speed': 1.0,
            'color_mode': 'dynamic'
        }
    
    def set_param(self, param_name: str, value: Any) -> None:
        """Set a parameter value"""
        if param_name in self.params:
            self.params[param_name] = value
    
    def update(self, audio_data=None, fft_data=None, beat_detected=False, frequency_bands=None) -> List[int]:
        """Update the LED colors based on audio data"""
        # This should be overridden by subclasses
        return self.flatten_colors()
    
    def get_led_colors(self) -> List[int]:
        """Get flattened LED colors as [r1, g1, b1, r2, g2, b2, ...]"""
        return self.flatten_colors()
    
    def flatten_colors(self) -> List[int]:
        """Convert LED colors to flat list for Arduino"""
        flattened = []
        for r, g, b in self.led_colors:
            flattened.extend([int(r), int(g), int(b)])
        return flattened
    
    def scale_color(self, color: Tuple[int, int, int], scale: float) -> Tuple[int, int, int]:
        """Scale a color by a factor"""
        scale = max(0.0, min(1.0, scale))
        r, g, b = color
        return (int(r * scale), int(g * scale), int(b * scale))
    
    def blend_colors(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int], 
                      blend_factor: float) -> Tuple[int, int, int]:
        """Blend two colors together"""
        blend_factor = max(0.0, min(1.0, blend_factor))
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        r = int(r1 * (1 - blend_factor) + r2 * blend_factor)
        g = int(g1 * (1 - blend_factor) + g2 * blend_factor)
        b = int(b1 * (1 - blend_factor) + b2 * blend_factor)
        return (r, g, b)
    
    def hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV to RGB color"""
        h = h % 360
        s = max(0.0, min(1.0, s))
        v = max(0.0, min(1.0, v))
        
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)
        
        return (r, g, b)


class BeatPulseVisualizer(BaseVisualizer):
    """Visualizer that pulses on detected beats"""
    
    def __init__(self, num_leds: int = NUM_LEDS):
        super().__init__(num_leds)
        self.hue = 0
        self.pulse_level = 0
        self.decay = 0.85  # How quickly pulse fades
        self.speed = 2     # How quickly hue changes
        
        # Predefine colors for different pulse intensities
        self.colors = [
            RED, BLUE, GREEN, PURPLE, YELLOW, CYAN
        ]
    
    def update(self, audio_data=None, fft_data=None, beat_detected=False, frequency_bands=None) -> List[int]:
        """Update visualization based on audio data and beat detection"""
        # Decay existing pulse
        self.pulse_level *= self.decay
        
        # If beat detected, set pulse to max
        if beat_detected:
            self.pulse_level = 1.0
            # Change color on beat
            self.hue = (self.hue + self.speed * 10) % 360
        
        # Slowly shift hue in any case
        self.hue = (self.hue + self.speed) % 360
        
        # Apply brightness from parameters
        brightness = self.params['brightness']
        sensitivity = self.params['sensitivity']
        
        # Create pulse effect
        if frequency_bands is not None and len(frequency_bands) > 0:
            # Use frequency information to influence color
            bass_level = np.mean(frequency_bands[:4]) if len(frequency_bands) >= 4 else 0
            mid_level = np.mean(frequency_bands[4:12]) if len(frequency_bands) >= 12 else 0
            high_level = np.mean(frequency_bands[12:]) if len(frequency_bands) >= 16 else 0
            
            # Normalize levels
            max_level = max(bass_level, mid_level, high_level, 1e-10)
            bass_norm = bass_level / max_level
            mid_norm = mid_level / max_level
            high_norm = high_level / max_level
            
            # Create color based on frequency distribution
            r = int(255 * bass_norm * brightness)
            g = int(255 * mid_norm * brightness)
            b = int(255 * high_norm * brightness)
            color = (r, g, b)
        else:
            # Use regular pulse color
            if self.params['color_mode'] == 'dynamic':
                color = self.hsv_to_rgb(self.hue, 1.0, brightness)
            else:
                # Use fixed color from the cycle
                color_idx = int(time.time() / 2) % len(self.colors)
                color = self.scale_color(self.colors[color_idx], brightness)
        
        # Apply pulse scaling
        pulse_color = self.scale_color(color, self.pulse_level * sensitivity)
        
        # Update all LEDs with pulse color
        self.led_colors = [pulse_color] * self.num_leds
        
        return self.flatten_colors()


class SpectrumVisualizer(BaseVisualizer):
    """Visualizer that shows frequency spectrum across the LED strip"""
    
    def __init__(self, num_leds: int = NUM_LEDS):
        super().__init__(num_leds)
        self.max_height = 0.0  # For normalization
        self.decay = 0.9       # Decay factor for max height
        self.smoothing = 0.7   # Smoothing factor for band transitions
        self.previous_bands = None
    
    def update(self, audio_data=None, fft_data=None, beat_detected=False, frequency_bands=None) -> List[int]:
        """Update visualization with frequency spectrum"""
        if frequency_bands is None or len(frequency_bands) == 0:
            return self.flatten_colors()
        
        # Get parameters
        brightness = self.params['brightness']
        sensitivity = self.params['sensitivity']
        
        # Apply smoothing to bands if we have previous data
        if self.previous_bands is not None:
            smoothed_bands = []
            for i, band in enumerate(frequency_bands):
                prev = self.previous_bands[i] if i < len(self.previous_bands) else 0
                smoothed = prev * self.smoothing + band * (1 - self.smoothing)
                smoothed_bands.append(smoothed)
            frequency_bands = smoothed_bands
        
        # Store for next update
        self.previous_bands = frequency_bands.copy()
        
        # Find max value for normalization
        current_max = np.max(frequency_bands)
        if current_max > self.max_height:
            self.max_height = current_max
        else:
            self.max_height = self.max_height * self.decay + current_max * (1 - self.decay)
        
        # Normalize bands
        if self.max_height > 0:
            normalized_bands = [min(1.0, b / self.max_height * sensitivity) for b in frequency_bands]
        else:
            normalized_bands = [0] * len(frequency_bands)
        
        # Map frequency bands to LEDs
        led_values = self.map_bands_to_leds(normalized_bands)
        
        # Apply colors based on frequency
        for i in range(self.num_leds):
            value = led_values[i]
            
            # Map each LED to a color based on position (frequency)
            hue = (i / self.num_leds) * 360  # Full color spectrum
            color = self.hsv_to_rgb(hue, 1.0, value * brightness)
            self.led_colors[i] = color
        
        return self.flatten_colors()
    
    def map_bands_to_leds(self, bands: List[float]) -> List[float]:
        """Map frequency bands to LED values using interpolation"""
        # Simple case: same number of bands as LEDs
        if len(bands) == self.num_leds:
            return bands
        
        # Need to interpolate
        led_values = [0] * self.num_leds
        
        # Use linear interpolation to map fewer bands to more LEDs
        if len(bands) < self.num_leds:
            for i in range(self.num_leds):
                band_idx = (i / self.num_leds) * (len(bands) - 1)
                band_idx_floor = int(band_idx)
                band_idx_ceil = min(band_idx_floor + 1, len(bands) - 1)
                blend = band_idx - band_idx_floor
                
                led_values[i] = bands[band_idx_floor] * (1 - blend) + bands[band_idx_ceil] * blend
        
        # Or average multiple bands to map to fewer LEDs
        else:
            for i in range(self.num_leds):
                start_band = int((i / self.num_leds) * len(bands))
                end_band = int(((i + 1) / self.num_leds) * len(bands))
                if start_band == end_band:
                    led_values[i] = bands[start_band]
                else:
                    led_values[i] = np.mean(bands[start_band:end_band])
        
        return led_values


class EnergyBeatVisualizer(BaseVisualizer):
    """Visualizer that reacts to both beats and audio energy"""
    
    def __init__(self, num_leds: int = NUM_LEDS):
        super().__init__(num_leds)
        self.energy = 0.0
        self.energy_decay = 0.9
        self.beat_decay = 0.7
        self.beat_intensity = 0.0
        self.base_hue = 0
        self.hue_shift_speed = 0.5
    
    def update(self, audio_data=None, fft_data=None, beat_detected=False, frequency_bands=None) -> List[int]:
        """Update visualization based on audio energy and beats"""
        # Get parameters
        brightness = self.params['brightness']
        sensitivity = self.params['sensitivity']
        
        # Calculate audio energy if we have audio data
        if audio_data is not None and len(audio_data) > 0:
            # Compute RMS energy
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            current_energy = np.sqrt(np.mean(audio_data ** 2)) * sensitivity
            self.energy = max(current_energy, self.energy * self.energy_decay)
        else:
            self.energy *= self.energy_decay
        
        # Update beat intensity
        if beat_detected:
            self.beat_intensity = 1.0
            # Change base hue on beat
            self.base_hue = (self.base_hue + 30) % 360
        else:
            self.beat_intensity *= self.beat_decay
        
        # Shift base hue slowly over time
        self.base_hue = (self.base_hue + self.hue_shift_speed) % 360
        
        # Create energy-based visualization
        for i in range(self.num_leds):
            # Position-based effects
            position = i / self.num_leds
            
            # Create different effects for different sections
            if position < 0.33:  # First third: bass responsive
                hue = (self.base_hue + 120) % 360
                intensity = self.energy * (1 - position * 2) + self.beat_intensity
            elif position < 0.67:  # Middle third: mid responsive
                hue = (self.base_hue + 240) % 360
                intensity = self.energy * (1 - abs(position - 0.5) * 4) + self.beat_intensity * 0.7
            else:  # Last third: high responsive
                hue = self.base_hue
                intensity = self.energy * (position * 2 - 1) + self.beat_intensity * 0.5
            
            # Apply color
            intensity = min(1.0, intensity) * brightness
            self.led_colors[i] = self.hsv_to_rgb(hue, 1.0, intensity)
        
        return self.flatten_colors()


# Register available visualizations
VISUALIZATIONS = {
    'beat_pulse': BeatPulseVisualizer,
    'spectrum': SpectrumVisualizer,
    'energy_beat': EnergyBeatVisualizer,
}

def create_visualizer(vis_type: str, num_leds: int = NUM_LEDS) -> BaseVisualizer:
    """Factory function to create a visualizer by name"""
    if vis_type not in VISUALIZATIONS:
        print(f"Warning: Unknown visualizer type '{vis_type}'. Using 'spectrum' instead.")
        vis_type = 'spectrum'
    
    return VISUALIZATIONS[vis_type](num_leds=num_leds)


if __name__ == "__main__":
    # Test code for visualizers
    print("Available visualizations:")
    for name in VISUALIZATIONS:
        print(f" - {name}")
    
    # Simple simulation with fake data
    vis = create_visualizer('spectrum')
    
    # Create fake frequency bands
    bands = np.zeros(16)
    for i in range(100):
        # Simulate changing bands
        for j in range(16):
            bands[j] = 0.5 + 0.5 * np.sin(i * 0.1 + j * 0.2)
        
        # Update and get colors
        colors = vis.update(frequency_bands=bands, beat_detected=(i % 10 == 0))
        print(f"Frame {i}: {len(colors) // 3} LEDs, beat: {(i % 10 == 0)}") 