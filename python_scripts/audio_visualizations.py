#!/usr/bin/env python3
"""
Audio Visualizations
Provides visualizations for LED strips based on audio input
"""
import time
import math
import numpy as np
from typing import List, Dict, Tuple, Union, Any, Optional
import random
import colorsys
from collections import deque

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
    """Base class for LED visualizers that process audio data"""
    
    def __init__(self, num_leds=NUM_LEDS):
        """Initialize visualizer with default settings"""
        self.num_leds = num_leds
        self.led_colors = [(0, 0, 0)] * num_leds  # Initialize with all LEDs off
        self.sensitivity = 1.0
        self.brightness = 1.0
        self.color_mode = "rainbow"  # rainbow, single, gradient
        self.primary_color = RED
        self.secondary_color = BLUE
        
    def update(self, audio_data):
        """Update the LED colors based on audio data"""
        # This should be implemented by subclasses
        pass
    
    def get_colors(self):
        """Return current LED colors as a list of (r,g,b) tuples"""
        return self.led_colors
    
    def get_led_colors(self):
        """Return current LED colors - compatibility method for web_visualizer"""
        return self.get_colors()
    
    def set_param(self, param, value):
        """Set a parameter of the visualizer"""
        if param == "sensitivity":
            self.sensitivity = max(0.1, min(5.0, float(value)))
        elif param == "brightness":
            self.brightness = max(0.1, min(1.0, float(value)))
        elif param == "color_mode":
            if value in ["rainbow", "single", "gradient"]:
                self.color_mode = value
        elif param == "primary_color":
            if isinstance(value, tuple) and len(value) == 3:
                self.primary_color = value
        elif param == "secondary_color":
            if isinstance(value, tuple) and len(value) == 3:
                self.secondary_color = value
    
    def _scale_color(self, color, scale_factor):
        """Scale a color by a factor (0.0 - 1.0)"""
        r, g, b = color
        factor = max(0.0, min(1.0, scale_factor))
        return (
            int(r * factor),
            int(g * factor),
            int(b * factor)
        )
    
    def _blend_colors(self, color1, color2, blend_factor):
        """Blend two colors with the given blend factor (0.0 - 1.0)"""
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        factor = max(0.0, min(1.0, blend_factor))
        return (
            int(r1 * (1 - factor) + r2 * factor),
            int(g1 * (1 - factor) + r2 * factor),
            int(b1 * (1 - factor) + b2 * factor)
        )
    
    def hsv_to_rgb(self, h, s, v):
        """Convert HSV (hue, saturation, value) to RGB color
        
        h: 0.0-1.0 (hue)
        s: 0.0-1.0 (saturation)
        v: 0.0-1.0 (value/brightness)
        
        Returns: tuple of (r, g, b) with values 0-255
        """
        h = max(0.0, min(1.0, h))
        s = max(0.0, min(1.0, s))
        v = max(0.0, min(1.0, v))
        
        if s == 0.0:
            # Gray scale
            r = g = b = int(v * 255)
            return (r, g, b)
        
        h *= 6  # Convert to sector between 0 and 6
        i = int(h)
        f = h - i  # Factorial part of h
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:  # i == 5
            r, g, b = v, p, q
        
        # Convert from 0-1 to 0-255
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        
        return (r, g, b)


class BeatPulseVisualizer(BaseVisualizer):
    """Visualizer that pulses on detected beats"""
    
    def __init__(self, num_leds: int = NUM_LEDS):
        super().__init__(num_leds)
        self.hue = 0.0  # Hue in range 0.0-1.0
        self.pulse_level = 0
        self.decay = 0.85  # How quickly pulse fades
        self.speed = 0.005  # How quickly hue changes
        
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
            self.hue = (self.hue + 0.1) % 1.0
        
        # Slowly shift hue in any case
        self.hue = (self.hue + self.speed) % 1.0
        
        # Apply brightness from parameters
        brightness = self.brightness
        sensitivity = self.sensitivity
        
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
            if self.color_mode == 'dynamic':
                color = self.hsv_to_rgb(self.hue, 1.0, brightness)
            else:
                # Use fixed color from the cycle
                color_idx = int(time.time() / 2) % len(self.colors)
                color = self._scale_color(self.colors[color_idx], brightness)
        
        # Apply pulse scaling
        pulse_color = self._scale_color(color, self.pulse_level * sensitivity)
        
        # Update all LEDs with pulse color
        self.led_colors = [pulse_color] * self.num_leds
        
        return self.get_colors()


class SpectrumVisualizer(BaseVisualizer):
    """Visualizer that shows frequency spectrum across the LED strip"""
    
    def __init__(self, num_leds: int = NUM_LEDS):
        super().__init__(num_leds)
        self.max_height = 0.0  # For normalization
        self.decay = 0.8       # Decay factor for max height
        self.smoothing = 0.9   # Smoothing factor for band transitions
        self.previous_bands = None
    
    def update(self, audio_data=None, fft_data=None, beat_detected=False, frequency_bands=None) -> List[int]:
        """Update visualization with frequency spectrum"""
        if frequency_bands is None or len(frequency_bands) == 0:
            return self.get_colors()
        
        # Get parameters
        brightness = self.brightness
        sensitivity = self.sensitivity
        
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
            hue = (i / self.num_leds)  # Full color spectrum (0.0 to 1.0)
            sat = 1.0  # Full saturation
            val = value * brightness  # Brightness based on audio level
            
            # Convert HSV to RGB - ensure all three RGB values are populated
            r, g, b = self.hsv_to_rgb(hue, sat, val)
            self.led_colors[i] = (r, g, b)
        
        return self.get_colors()
    
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
        self.base_hue = 0.0  # Hue in range 0.0-1.0
        self.hue_shift_speed = 0.001
    
    def update(self, audio_data=None, fft_data=None, beat_detected=False, frequency_bands=None) -> List[int]:
        """Update visualization based on audio energy and beats"""
        # Get parameters
        brightness = self.brightness
        sensitivity = self.sensitivity
        
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
            self.base_hue = (self.base_hue + 0.1) % 1.0
        else:
            self.beat_intensity *= self.beat_decay
        
        # Shift base hue slowly over time
        self.base_hue = (self.base_hue + self.hue_shift_speed) % 1.0
        
        # Create energy-based visualization
        for i in range(self.num_leds):
            # Position-based effects
            position = i / self.num_leds
            
            # Create different effects for different sections
            if position < 0.33:  # First third: bass responsive
                hue = (self.base_hue + 0.33) % 1.0
                intensity = self.energy * (1 - position * 2) + self.beat_intensity
            elif position < 0.67:  # Middle third: mid responsive
                hue = (self.base_hue + 0.67) % 1.0
                intensity = self.energy * (1 - abs(position - 0.5) * 4) + self.beat_intensity * 0.7
            else:  # Last third: high responsive
                hue = self.base_hue
                intensity = self.energy * (position * 2 - 1) + self.beat_intensity * 0.5
            
            # Apply color
            intensity = min(1.0, intensity) * brightness
            self.led_colors[i] = self.hsv_to_rgb(hue, 1.0, intensity)
        
        return self.get_colors()


class BassImpactVisualizer(BaseVisualizer):
    """Visualizer that emphasizes bass frequencies with powerful visual impact"""
    
    def __init__(self, num_leds: int = NUM_LEDS):
        super().__init__(num_leds)
        # Bass tracking
        self.bass_energy = 0.0
        self.bass_decay = 0.8
        self.bass_peak = 0.0
        self.peak_decay = 0.95
        
        # Mid/high tracking for contrast
        self.mid_energy = 0.0
        self.high_energy = 0.0
        self.mid_decay = 0.7
        self.high_decay = 0.6
        
        # Center-out propagation system
        self.propagation_waves = []  # Store active propagation waves
        
        # Beat tracking
        self.beat_cooldown = 0  # Frames before next beat can be registered
        
        # Colors
        self.bass_color = (255, 0, 0)  # Red for bass impact
        self.mid_color = (0, 255, 0)   # Green for mids
        self.high_color = (0, 80, 255)  # Blue for highs
        
        # Previous frame data for smoothing
        self.previous_colors = [(0, 0, 0)] * num_leds
    
    def update(self, audio_data=None, fft_data=None, beat_detected=False, frequency_bands=None) -> List[int]:
        """Update visualization based on frequency bands with bass emphasis"""
        # Reset LED colors
        new_colors = [(0, 0, 0)] * self.num_leds
        
        # Get parameters
        brightness = self.brightness
        sensitivity = self.sensitivity
        
        # Extract and update frequency band energies
        if frequency_bands is not None and len(frequency_bands) >= 7:
            # Use first two bands for bass (usually 20-250Hz)
            current_bass = np.mean(frequency_bands[:2]) * sensitivity
            
            # Use middle bands for mids (usually 250-4000Hz)
            current_mid = np.mean(frequency_bands[2:5]) * sensitivity
            
            # Use highest bands for highs (usually 4000-20000Hz)
            current_high = np.mean(frequency_bands[5:]) * sensitivity
            
            # Update running averages with decay
            self.bass_energy = max(current_bass, self.bass_energy * self.bass_decay)
            self.mid_energy = max(current_mid, self.mid_energy * self.mid_decay)
            self.high_energy = max(current_high, self.high_energy * self.high_decay)
            
            # Track peak bass for normalization
            if self.bass_energy > self.bass_peak:
                self.bass_peak = self.bass_energy
            else:
                self.bass_peak = self.bass_peak * self.peak_decay + self.bass_energy * (1 - self.peak_decay)
        else:
            # Decay if no data
            self.bass_energy *= self.bass_decay
            self.mid_energy *= self.mid_decay
            self.high_energy *= self.high_decay
        
        # Normalize bass energy (0.0-1.0)
        if self.bass_peak > 0:
            normalized_bass = min(1.0, self.bass_energy / self.bass_peak)
        else:
            normalized_bass = 0
        
        # Generate a center-out propagation wave on strong bass or beats
        if beat_detected and self.beat_cooldown <= 0:
            # Add a new propagation wave from the center
            center = self.num_leds // 2
            wave_strength = normalized_bass * 1.2  # Slightly boost it
            self.propagation_waves.append({
                'center': center,
                'strength': wave_strength,
                'width': 1,  # Start with a small width
                'speed': 1 + normalized_bass * 2,  # Speed based on bass intensity
                'age': 0     # New wave
            })
            self.beat_cooldown = 5  # Don't trigger too frequently
        elif normalized_bass > 0.7 and self.beat_cooldown <= 0:
            # Strong bass hit also creates waves
            center = self.num_leds // 2
            wave_strength = normalized_bass
            self.propagation_waves.append({
                'center': center,
                'strength': wave_strength,
                'width': 1,
                'speed': 1 + normalized_bass,
                'age': 0
            })
            self.beat_cooldown = 3  # Shorter cooldown for bass-triggered waves
        
        # Decrease beat cooldown
        if self.beat_cooldown > 0:
            self.beat_cooldown -= 1
        
        # Process all active waves
        active_waves = []
        for wave in self.propagation_waves:
            # Expand the wave
            wave['width'] += wave['speed']
            wave['age'] += 1
            
            # Calculate wave start and end positions
            half_width = wave['width'] // 2
            start = max(0, int(wave['center'] - half_width))
            end = min(self.num_leds - 1, int(wave['center'] + half_width))
            
            # Calculate intensity decay factor (waves fade as they expand)
            fade_factor = max(0, 1.0 - (wave['age'] / 20.0))
            
            # Apply wave to LEDs
            for i in range(start, end + 1):
                distance = abs(i - wave['center'])
                intensity = max(0, 1.0 - (distance / half_width)) if half_width > 0 else 0
                
                # Scale wave intensity
                wave_intensity = intensity * wave['strength'] * fade_factor
                
                # Base colors on frequency energy
                r = int(self.bass_color[0] * wave_intensity)
                g = int(self.mid_color[1] * wave_intensity * (self.mid_energy / max(self.bass_energy, 0.01)))
                b = int(self.high_color[2] * wave_intensity * (self.high_energy / max(self.bass_energy, 0.01)))
                
                # Add to existing color (allow overlapping waves)
                new_r = min(255, new_colors[i][0] + r)
                new_g = min(255, new_colors[i][1] + g)
                new_b = min(255, new_colors[i][2] + b)
                
                new_colors[i] = (new_r, new_g, new_b)
            
            # Keep active waves that haven't expanded beyond the strip
            if wave['age'] < 20 and fade_factor > 0.1:
                active_waves.append(wave)
        
        # Update active waves list
        self.propagation_waves = active_waves
        
        # Add a persistent bass indicator at the center
        center_range = self.num_leds // 5  # 20% of LEDs in the center
        center_start = (self.num_leds - center_range) // 2
        center_end = center_start + center_range
        
        for i in range(center_start, center_end):
            # Calculate position within center segment (0.0-1.0)
            pos = (i - center_start) / center_range
            
            # Create pulse effect at center that responds to bass
            pulse_intensity = normalized_bass * (1.0 - abs(pos - 0.5) * 2)
            
            # Create color based on bass intensity
            r = int(self.bass_color[0] * pulse_intensity)
            g = int(self.mid_color[1] * self.mid_energy / max(1.0, self.bass_peak) * pulse_intensity)
            b = int(self.high_color[2] * self.high_energy / max(1.0, self.bass_peak) * pulse_intensity)
            
            # Add to existing color
            new_r = min(255, new_colors[i][0] + r)
            new_g = min(255, new_colors[i][1] + g)
            new_b = min(255, new_colors[i][2] + b)
            
            new_colors[i] = (new_r, new_g, new_b)
        
        # Apply smoothing with previous frame
        for i in range(self.num_leds):
            self.led_colors[i] = self._blend_colors(
                self.previous_colors[i],
                new_colors[i],
                0.6  # Blend factor: higher = more responsive, lower = smoother
            )
        
        # Store colors for next frame
        self.previous_colors = new_colors.copy()
        
        # Apply global brightness
        if brightness < 1.0:
            for i in range(self.num_leds):
                self.led_colors[i] = self._scale_color(self.led_colors[i], brightness)
        
        return self.get_colors()


class FrequencyBarsVisualizer(BaseVisualizer):
    """Visualizer that divides the LED strip into frequency bands"""
    
    def __init__(self, num_leds: int = NUM_LEDS):
        super().__init__(num_leds)
        # Default number of frequency bands to display
        self.num_bands = 5
        # Map of bands to colors
        self.band_colors = [
            (255, 0, 0),    # Bass - Red
            (255, 165, 0),  # Low-mid - Orange
            (255, 255, 0),  # Mid - Yellow 
            (0, 255, 0),    # High-mid - Green
            (0, 0, 255)     # Treble - Blue
        ]
        # Default to rainbow colors
        self.use_rainbow = True
        # LED sections for each band
        self.sections = self._calculate_sections()
        # Previous band values for smoothing
        self.prev_band_values = [0.0] * self.num_bands
        # Smoothing factor (0-1), higher = more smoothing
        self.smoothing = 0.7
    
    def _calculate_sections(self):
        """Calculate which LEDs correspond to which frequency band"""
        sections = []
        leds_per_band = self.num_leds // self.num_bands
        remaining = self.num_leds % self.num_bands
        
        start_idx = 0
        for i in range(self.num_bands):
            # Distribute remaining LEDs among the first bands
            extra = 1 if i < remaining else 0
            section_size = leds_per_band + extra
            sections.append((start_idx, start_idx + section_size))
            start_idx += section_size
            
        return sections
    
    def update(self, audio_data=None, fft_data=None, beat_detected=False, frequency_bands=None) -> List[int]:
        """Update visualization based on frequency bands data"""
        if frequency_bands is None or len(frequency_bands) == 0:
            return self.get_colors()
            
        # Initialize all LEDs to off
        self.led_colors = [(0, 0, 0)] * self.num_leds
            
        # Get the number of bands we can use
        available_bands = min(len(frequency_bands), self.num_bands)
        
        # Extract and scale the bands we need
        band_values = []
        for i in range(available_bands):
            # For each visualization band, take a group of input frequency bands
            band_start = i * len(frequency_bands) // available_bands
            band_end = (i + 1) * len(frequency_bands) // available_bands
            
            # Take the average value for this group of frequency bands
            if band_start < band_end:
                value = np.mean(frequency_bands[band_start:band_end])
            else:
                value = frequency_bands[band_start]
                
            # Apply sensitivity
            value = value * self.sensitivity
            
            # Apply smoothing with previous values
            if i < len(self.prev_band_values):
                value = (value * (1 - self.smoothing)) + (self.prev_band_values[i] * self.smoothing)
                self.prev_band_values[i] = value
            else:
                self.prev_band_values.append(value)
                
            # Cap at 1.0
            value = min(1.0, value)
            band_values.append(value)
            
        # Fill in missing band values if needed
        while len(band_values) < self.num_bands:
            band_values.append(0.0)
            
        # Apply band values to LED sections
        for i, (start_idx, end_idx) in enumerate(self.sections):
            if i >= len(band_values):
                continue
                
            # Calculate how many LEDs to light in this section
            level = band_values[i]
            section_size = end_idx - start_idx
            leds_on = int(section_size * level)
            
            # Get color for this band
            if self.use_rainbow:
                # Use rainbow coloring (hue based on band index)
                hue = i / self.num_bands
                color = self.hsv_to_rgb(hue, 1.0, self.brightness)
            else:
                # Use predefined colors
                if i < len(self.band_colors):
                    color = self.band_colors[i]
                else:
                    color = WHITE
                # Apply brightness
                color = self._scale_color(color, self.brightness)
                
            # Set the LEDs for this band
            for j in range(leds_on):
                if start_idx + j < self.num_leds:
                    # Create a gradient effect within each band
                    brightness_factor = (j + 1) / section_size
                    led_color = self._scale_color(color, brightness_factor)
                    self.led_colors[start_idx + j] = led_color
                    
        return self.get_colors()
    
    def set_param(self, param, value):
        """Set visualizer parameters"""
        super().set_param(param, value)
        
        if param == "use_rainbow":
            self.use_rainbow = bool(value)
        elif param == "num_bands":
            try:
                new_bands = int(value)
                if 2 <= new_bands <= 10:  # Reasonable limits
                    self.num_bands = new_bands
                    self.sections = self._calculate_sections()
                    self.prev_band_values = [0.0] * self.num_bands
            except ValueError:
                pass
        elif param == "smoothing":
            try:
                value = float(value)
                self.smoothing = max(0.0, min(0.95, value))
            except ValueError:
                pass


class VUMeterVisualizer(BaseVisualizer):
    """Visualizer that mimics a VU meter (volume unit meter)"""
    
    def __init__(self, num_leds: int = NUM_LEDS):
        super().__init__(num_leds)
        # Decay rate for the meter
        self.decay_rate = 0.05
        # Current meter level (0.0 - 1.0)
        self.level = 0.0
        # Peak hold value
        self.peak_level = 0.0
        # How fast the peak decays
        self.peak_decay = 0.01
        # Whether to show the peak indicator
        self.show_peak = True
        # VU meter color gradient: green -> yellow -> red
        self.start_color = (0, 255, 0)  # Green
        self.mid_color = (255, 255, 0)  # Yellow
        self.end_color = (255, 0, 0)    # Red
        # Level at which yellow begins
        self.yellow_threshold = 0.5
        # Level at which red begins
        self.red_threshold = 0.8
        # Attack rate - how quickly it responds to increases
        self.attack_rate = 0.5
    
    def update(self, audio_data=None, fft_data=None, beat_detected=False, frequency_bands=None) -> List[int]:
        """Update visualization based on audio volume"""
        # Calculate current audio level
        current_level = 0.0
        
        if audio_data is not None and len(audio_data) > 0:
            # Calculate RMS (root mean square) of audio data
            audio_data = np.array(audio_data)
            rms = np.sqrt(np.mean(np.square(audio_data)))
            # Scale and limit
            current_level = min(1.0, rms * 3.0 * self.sensitivity)
        elif frequency_bands is not None and len(frequency_bands) > 0:
            # Use overall energy from frequency bands
            current_level = min(1.0, np.mean(frequency_bands) * self.sensitivity)
        
        # Apply attack/decay behavior
        if current_level > self.level:
            # Fast attack
            self.level = self.level + (current_level - self.level) * self.attack_rate
        else:
            # Slow decay
            self.level = max(0.0, self.level - self.decay_rate)
        
        # Update peak level
        if self.level > self.peak_level:
            self.peak_level = self.level
        else:
            self.peak_level = max(0.0, self.peak_level - self.peak_decay)
        
        # Map the level to LEDs
        level_leds = int(self.level * self.num_leds)
        peak_led = int(self.peak_level * self.num_leds)
        
        # Initialize all LEDs to off
        self.led_colors = [(0, 0, 0)] * self.num_leds
        
        # Set the colors for the active LED range
        for i in range(level_leds):
            position = i / (self.num_leds - 1)
            
            # Apply color based on position in meter
            if position < self.yellow_threshold:
                # Green to yellow gradient
                color_factor = position / self.yellow_threshold
                color = self._blend_colors(self.start_color, self.mid_color, color_factor)
            elif position < self.red_threshold:
                # Yellow to red gradient
                color_factor = (position - self.yellow_threshold) / (self.red_threshold - self.yellow_threshold)
                color = self._blend_colors(self.mid_color, self.end_color, color_factor)
            else:
                # Red
                color = self.end_color
            
            # Apply brightness
            color = self._scale_color(color, self.brightness)
            self.led_colors[i] = color
        
        # Draw peak indicator if enabled
        if self.show_peak and peak_led > 0 and peak_led < self.num_leds:
            self.led_colors[peak_led - 1] = self._scale_color(WHITE, self.brightness)
        
        return self.get_colors()
    
    def set_param(self, param, value):
        """Set visualizer parameters"""
        super().set_param(param, value)
        
        if param == "decay_rate":
            try:
                value = float(value)
                self.decay_rate = max(0.01, min(0.2, value))
            except ValueError:
                pass
        elif param == "peak_decay":
            try:
                value = float(value)
                self.peak_decay = max(0.001, min(0.05, value))
            except ValueError:
                pass
        elif param == "show_peak":
            self.show_peak = bool(value)
        elif param == "attack_rate":
            try:
                value = float(value)
                self.attack_rate = max(0.1, min(1.0, value))
            except ValueError:
                pass
        elif param == "yellow_threshold":
            try:
                value = float(value)
                self.yellow_threshold = max(0.2, min(0.7, value))
            except ValueError:
                pass
        elif param == "red_threshold":
            try:
                value = float(value)
                self.red_threshold = max(self.yellow_threshold + 0.1, min(0.95, value))
            except ValueError:
                pass


# Add new class for performance benchmarking visualization
class PerfBenchmarkVisualizer(BaseVisualizer):
    """Performance benchmark visualization that clearly shows accuracy and frame rate"""
    
    def __init__(self, num_leds=60):
        super().__init__(num_leds)
        self.tick_position = 0
        self.tick_color = (255, 255, 255)  # White tick mark
        self.start_time = time.time()
        self.frame_counter = 0
        self.expected_frame_counter = 0
        self.expected_fps = 120  # Target FPS
        self.last_reset = time.time()
        self.background_hue = 0.0
    
    def update(self, audio_data=None, fft_data=None, beat_detected=False, frequency_bands=None):
        # Increment frame counter
        self.frame_counter += 1
        
        # Calculate expected frames based on elapsed time and target FPS
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.expected_frame_counter = int(elapsed * self.expected_fps)
        
        # Every second, update metrics and reset counters if needed
        if current_time - self.last_reset >= 1.0:
            frame_diff = abs(self.frame_counter - self.expected_frame_counter)
            self.accuracy = 100 - min(100, (frame_diff / self.expected_fps) * 100)
            self.last_reset = current_time
        
        # Clear all LEDs
        self.led_colors = [(0, 0, 0)] * self.num_leds
        
        # Create a moving tick mark that should move at exactly 1 position per frame at 60fps
        # This will visually show if we're dropping frames (it would move inconsistently)
        self.tick_position = (self.tick_position + 1) % self.num_leds
        self.led_colors[self.tick_position] = self.tick_color
        
        # Create visual indicator of frame accuracy (color-coded background)
        # Blue = Perfect performance, Green = Good, Yellow = Fair, Red = Poor
        self.background_hue = (self.background_hue + 0.01) % 1.0
        background_color = self.hsv_to_rgb(self.background_hue, 1.0, 0.2)  # Low brightness background
        
        # Fill in background on the LEDs that aren't the tick mark
        for i in range(self.num_leds):
            if i != self.tick_position:
                self.led_colors[i] = background_color
                
        # Create pattern that would reveal timing issues
        # Every 15 frames, add a bright flash that should be perfectly synchronized
        if self.frame_counter % 15 == 0:
            marker_position = (self.tick_position + self.num_leds // 2) % self.num_leds
            self.led_colors[marker_position] = (255, 255, 0)  # Yellow marker
            
        # Add position markers at regular intervals
        for i in range(0, self.num_leds, 10):
            if i != self.tick_position:
                # Gradual color change based on position
                position_hue = i / self.num_leds
                position_color = self.hsv_to_rgb(position_hue, 1.0, 0.5)
                self.led_colors[i] = position_color
    
    def hsv_to_rgb(self, h, s, v):
        """Convert HSV color to RGB"""
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))


# Register available visualizations
VISUALIZATIONS = {
    'beat_pulse': BeatPulseVisualizer,
    'spectrum': SpectrumVisualizer,
    'energy_beat': EnergyBeatVisualizer,
    'bass_impact': BassImpactVisualizer,
    'frequency_bars': FrequencyBarsVisualizer,
    'vu_meter': VUMeterVisualizer,
    'perf_benchmark': PerfBenchmarkVisualizer,
}

def create_visualizer(vis_type: str, num_leds: int = NUM_LEDS) -> BaseVisualizer:
    """Factory function to create visualizer by type name"""
    vis_type = vis_type.lower()
    
    if vis_type == "beat_pulse":
        return BeatPulseVisualizer(num_leds)
    elif vis_type == "spectrum":
        return SpectrumVisualizer(num_leds)
    elif vis_type == "energy_beat":
        return EnergyBeatVisualizer(num_leds)
    elif vis_type == "bass_impact":
        return BassImpactVisualizer(num_leds)
    elif vis_type == "frequency_bars":
        return FrequencyBarsVisualizer(num_leds)
    elif vis_type == "vu_meter":
        return VUMeterVisualizer(num_leds)
    elif vis_type == "perf_benchmark":
        return PerfBenchmarkVisualizer(num_leds)
    else:
        # Default to spectrum visualizer
        return SpectrumVisualizer(num_leds)


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