#!/usr/bin/env python3
"""
Audio Visualizations Module
Provides different visualizations for audio data on LED strips
"""
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# Constants
NUM_LEDS = 60

# Color definitions
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)
CYAN = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Base colors for visualizations
RAINBOW_COLORS = [RED, (255, 127, 0), YELLOW, GREEN, BLUE, (75, 0, 130), (148, 0, 211)]


class BaseVisualizer:
    """Base class for all audio visualizations"""
    
    def __init__(self, num_leds: int = NUM_LEDS):
        """
        Initialize base visualizer
        
        Args:
            num_leds: Number of LEDs in the strip
        """
        self.num_leds = num_leds
        self.colors = [(0, 0, 0)] * num_leds  # Start with all LEDs off
        self.params = {
            'brightness': 1.0,      # Global brightness multiplier
            'sensitivity': 1.0,     # Sensitivity to audio input
            'color_scheme': 0,      # Index of color scheme to use
            'speed': 1.0,           # Animation speed multiplier
            'decay': 0.8,           # How quickly effects fade (0-1)
        }
    
    def update(self, is_beat: bool, fft_data: np.ndarray, bands: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Update visualization based on audio data
        
        Args:
            is_beat: Whether a beat was detected
            fft_data: FFT data
            bands: Frequency band energies
            
        Returns:
            List of RGB tuples for each LED
        """
        # Implement in subclasses
        return [(0, 0, 0)] * self.num_leds
    
    def set_param(self, param_name: str, value: float) -> None:
        """Set a visualization parameter"""
        if param_name in self.params:
            self.params[param_name] = value
    
    def get_colors(self) -> List[Tuple[int, int, int]]:
        """Get the current colors for the LED strip"""
        # Apply global brightness
        brightness = self.params['brightness']
        if brightness < 1.0:
            return [self._scale_color(color, brightness) for color in self.colors]
        return self.colors
    
    @staticmethod
    def _scale_color(color: Tuple[int, int, int], scale: float) -> Tuple[int, int, int]:
        """Scale RGB color by a factor"""
        return (
            int(color[0] * scale),
            int(color[1] * scale),
            int(color[2] * scale)
        )
    
    @staticmethod
    def _blend_colors(color1: Tuple[int, int, int], color2: Tuple[int, int, int], 
                     ratio: float) -> Tuple[int, int, int]:
        """Blend two colors with the given ratio (0.0 = color1, 1.0 = color2)"""
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        return (r, g, b)
    
    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV color to RGB"""
        h = h % 360
        h_i = int(h / 60)
        f = h / 60 - h_i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        if h_i == 0:
            r, g, b = v, t, p
        elif h_i == 1:
            r, g, b = q, v, p
        elif h_i == 2:
            r, g, b = p, v, t
        elif h_i == 3:
            r, g, b = p, q, v
        elif h_i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        
        return (int(r * 255), int(g * 255), int(b * 255))


class BeatPulseVisualizer(BaseVisualizer):
    """Visualization that pulses with the beat"""
    
    def __init__(self, num_leds: int = NUM_LEDS):
        super().__init__(num_leds)
        
        # Additional parameters
        self.params.update({
            'pulse_length': 0.7,    # Length of pulse (0-1)
            'color_variation': 0.3,  # How much to vary color (0-1)
        })
        
        # State variables
        self.pulse_intensity = 0.0
        self.color_offset = 0.0
        self.base_color_index = 0
        self.color_schemes = [
            [RED, PURPLE, RED, PURPLE],                      # Red-Purple
            [BLUE, CYAN, BLUE, CYAN],                    # Blue-Cyan
            [GREEN, YELLOW, GREEN, YELLOW],              # Green-Yellow
            RAINBOW_COLORS,                              # Rainbow
            [WHITE, (200, 200, 200), WHITE, (200, 200, 200)]  # White
        ]
    
    def update(self, is_beat: bool, fft_data: np.ndarray, bands: np.ndarray) -> List[Tuple[int, int, int]]:
        """Update visualization based on audio data"""
        # Get parameters
        sensitivity = self.params['sensitivity']
        decay = self.params['decay']
        pulse_length = self.params['pulse_length']
        color_scheme = int(min(self.params['color_scheme'], len(self.color_schemes) - 1))
        
        # Update state based on beat
        if is_beat:
            # Set pulse to max intensity
            self.pulse_intensity = 1.0
            
            # Change base color on beat occasionally
            if random.random() < 0.3:
                self.base_color_index = (self.base_color_index + 1) % len(self.color_schemes[color_scheme])
        else:
            # Decay pulse
            self.pulse_intensity *= decay
        
        # Get base color
        base_color = self.color_schemes[color_scheme][self.base_color_index]
        
        # Calculate colors for each LED
        new_colors = []
        for i in range(self.num_leds):
            # Calculate position in the pulse (0-1)
            pos = float(i) / self.num_leds
            
            # Only light LEDs within the pulse length
            in_pulse = pos < pulse_length
            
            if in_pulse:
                # Normalize position within pulse
                pulse_pos = pos / pulse_length
                
                # Calculate intensity at this position (center is brightest)
                position_intensity = 1.0 - abs(2 * pulse_pos - 1)
                
                # Apply beat intensity
                led_intensity = position_intensity * self.pulse_intensity * sensitivity
                
                # Small color variation based on position
                hue_offset = self.params['color_variation'] * position_intensity * 30
                if isinstance(base_color, tuple):
                    # Convert to HSV, adjust hue, convert back to RGB
                    r, g, b = base_color
                    h, s, v = self._rgb_to_hsv(r, g, b)
                    h = (h + hue_offset) % 360
                    color = self._hsv_to_rgb(h, s, v)
                else:
                    color = base_color
                
                # Scale by intensity
                color = self._scale_color(color, led_intensity)
            else:
                color = BLACK
            
            new_colors.append(color)
        
        self.colors = new_colors
        return self.colors
    
    @staticmethod
    def _rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
        """Convert RGB to HSV color space"""
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g - b) / df) + 360) % 360
        elif mx == g:
            h = (60 * ((b - r) / df) + 120) % 360
        elif mx == b:
            h = (60 * ((r - g) / df) + 240) % 360
        if mx == 0:
            s = 0
        else:
            s = df / mx
        v = mx
        return (h, s, v)


class SpectrumVisualizer(BaseVisualizer):
    """Visualization that shows audio frequency spectrum"""
    
    def __init__(self, num_leds: int = NUM_LEDS):
        super().__init__(num_leds)
        
        # Additional parameters
        self.params.update({
            'smoothing': 0.7,      # Smoothing factor between frames (0-1)
            'baseline': 0.1,       # Baseline brightness when no audio
            'normalize': True,     # Whether to normalize the spectrum
        })
        
        # Previous band values for smoothing
        self.prev_values = np.zeros(num_leds)
        
        # Color gradient for visualization
        self.color_schemes = [
            [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)],  # Blue to Red
            [(0, 0, 255), (255, 0, 255), (255, 0, 0)],  # Blue to Purple to Red
            [(0, 255, 0), (255, 255, 0), (255, 0, 0)],  # Green to Yellow to Red
            RAINBOW_COLORS,  # Rainbow
            [(0, 0, 255), (255, 255, 255)]  # Blue to White
        ]
    
    def update(self, is_beat: bool, fft_data: np.ndarray, bands: np.ndarray) -> List[Tuple[int, int, int]]:
        """Update visualization based on audio data"""
        # Get parameters
        sensitivity = self.params['sensitivity']
        smoothing = self.params['smoothing']
        baseline = self.params['baseline']
        color_scheme_idx = int(min(self.params['color_scheme'], len(self.color_schemes) - 1))
        color_scheme = self.color_schemes[color_scheme_idx]
        
        # Normalize bands data to 0-1 range
        if len(bands) == 0:
            normalized_bands = np.zeros(self.num_leds)
        else:
            # Apply sensitivity
            scaled_bands = bands * sensitivity
            
            # Normalize if needed
            if self.params['normalize'] and np.max(scaled_bands) > 0:
                normalized_bands = scaled_bands / np.max(scaled_bands)
            else:
                normalized_bands = np.clip(scaled_bands, 0, 1)
            
            # Add baseline
            normalized_bands = baseline + (1 - baseline) * normalized_bands
            
            # Resize to match LED count
            if len(normalized_bands) != self.num_leds:
                # Simple linear interpolation
                old_indices = np.linspace(0, len(normalized_bands) - 1, len(normalized_bands))
                new_indices = np.linspace(0, len(normalized_bands) - 1, self.num_leds)
                normalized_bands = np.interp(new_indices, old_indices, normalized_bands)
        
        # Apply smoothing with previous values
        smoothed_values = smoothing * self.prev_values + (1 - smoothing) * normalized_bands
        self.prev_values = smoothed_values
        
        # Map values to colors
        new_colors = []
        for i, value in enumerate(smoothed_values):
            # Map value to position in color gradient
            if value <= 0:
                color = BLACK
            else:
                # Map value to position in color gradient
                color_pos = value
                color_idx = color_pos * (len(color_scheme) - 1)
                idx1 = int(color_idx)
                idx2 = min(idx1 + 1, len(color_scheme) - 1)
                color_ratio = color_idx - idx1
                
                # Blend between the two nearest colors
                color = self._blend_colors(
                    color_scheme[idx1],
                    color_scheme[idx2],
                    color_ratio
                )
            
            new_colors.append(color)
        
        self.colors = new_colors
        return self.colors


class EnergyBeatVisualizer(BaseVisualizer):
    """Visualization that reacts to both beats and energy levels"""
    
    def __init__(self, num_leds: int = NUM_LEDS):
        super().__init__(num_leds)
        
        # Additional parameters
        self.params.update({
            'energy_scale': 1.5,    # Scaling factor for energy response
            'beat_intensity': 2.0,  # How much to amplify on beat
            'color_speed': 0.2,     # Speed of color cycling
        })
        
        # State variables
        self.energy_levels = np.zeros(self.num_leds)
        self.hue_offset = 0.0
        self.beat_time = 0.0
    
    def update(self, is_beat: bool, fft_data: np.ndarray, bands: np.ndarray) -> List[Tuple[int, int, int]]:
        """Update visualization based on audio data"""
        # Get parameters
        sensitivity = self.params['sensitivity']
        decay = self.params['decay']
        energy_scale = self.params['energy_scale']
        beat_intensity = self.params['beat_intensity']
        color_speed = self.params['color_speed']
        
        # Calculate overall energy from bands
        if len(bands) > 0:
            # Get average energy across all bands
            avg_energy = np.mean(bands) * sensitivity * energy_scale
            
            # If beat detected, amplify energy
            if is_beat:
                avg_energy *= beat_intensity
                self.beat_time = 1.0
            else:
                self.beat_time *= decay
                
            # Shift energy levels
            self.energy_levels = np.roll(self.energy_levels, 1)
            
            # Set new energy at the center
            center = self.num_leds // 2
            self.energy_levels[center] = avg_energy
            
            # Smooth the energy levels (moving average)
            kernel = np.array([0.25, 0.5, 0.25])
            for i in range(1, self.num_leds - 1):
                # Apply smoothing kernel
                self.energy_levels[i] = (
                    self.energy_levels[i-1] * kernel[0] +
                    self.energy_levels[i] * kernel[1] +
                    self.energy_levels[i+1] * kernel[2]
                )
            
            # Apply decay to all values
            self.energy_levels *= decay
        
        # Update hue offset
        self.hue_offset = (self.hue_offset + color_speed) % 360
        
        # Create colors for each LED
        new_colors = []
        for i, energy in enumerate(self.energy_levels):
            # Map position to base hue (0-360)
            pos_hue = (i * 360 / self.num_leds + self.hue_offset) % 360
            
            # Saturation and value based on energy
            saturation = min(1.0, 0.3 + energy * 0.7)
            value = min(1.0, energy)
            
            # Boost on beat
            if self.beat_time > 0:
                value = min(1.0, value + self.beat_time * 0.5)
            
            # Convert to RGB
            color = self._hsv_to_rgb(pos_hue, saturation, value)
            new_colors.append(color)
        
        self.colors = new_colors
        return self.colors


# Dictionary mapping visualization names to their classes
VISUALIZATIONS = {
    'beat_pulse': BeatPulseVisualizer,
    'spectrum': SpectrumVisualizer,
    'energy_beat': EnergyBeatVisualizer,
}


def create_visualizer(name: str, num_leds: int = NUM_LEDS) -> BaseVisualizer:
    """Factory function to create a visualizer by name"""
    if name in VISUALIZATIONS:
        return VISUALIZATIONS[name](num_leds)
    else:
        # Default to beat pulse
        print(f"Visualization '{name}' not found, using beat_pulse")
        return BeatPulseVisualizer(num_leds)


if __name__ == "__main__":
    # Demo visualization with random data
    from time import sleep
    
    # Create visualizer
    visualizer = create_visualizer('beat_pulse')
    
    # Generate some fake data
    for i in range(100):
        # Fake a beat every 0.5 seconds
        is_beat = (i % 5) == 0
        
        # Fake FFT data
        fft_data = np.random.rand(512) * (0.5 + 0.5 * (is_beat))
        
        # Fake frequency bands
        bands = np.array([
            np.random.rand() * (0.5 + 0.5 * (is_beat and j % 3 == 0))
            for j in range(8)
        ])
        
        # Update visualizer
        colors = visualizer.update(is_beat, fft_data, bands)
        
        # Print a simple visualization
        intensity = sum(sum(c) for c in colors) / (NUM_LEDS * 3 * 255)
        bar_length = int(intensity * 50)
        print(f"{'#' * bar_length}{' ' * (50 - bar_length)} | Beat: {is_beat}")
        
        sleep(0.1) 