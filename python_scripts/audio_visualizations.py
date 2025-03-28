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
        self.color_scheme = "rainbow"  # rainbow, fire, ocean, forest, neon, pastels
        self.response_speed = 0.5  # 0.0 to 1.0, controls speed of response
        self.smoothing_enabled = True  # Whether to use smoothing
        
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
        elif param == "color_scheme":
            valid_schemes = ["rainbow", "fire", "ocean", "forest", "neon", "pastels"]
            if value in valid_schemes:
                self.color_scheme = value
        elif param == "response_speed":
            self.response_speed = max(0.0, min(1.0, float(value)))
        elif param == "smoothing_enabled":
            self.smoothing_enabled = bool(value)
    
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
            int(g1 * (1 - factor) + g2 * factor),
            int(b1 * (1 - factor) + b2 * factor)
        )
    
    def _hsv_to_rgb(self, h, s, v):
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

    def _get_color_from_scheme(self, position, intensity=1.0):
        """Get color from current color scheme based on position (0.0 - 1.0)"""
        if self.color_scheme == "rainbow":
            return self._hsv_to_rgb(position, 1.0, intensity)
        elif self.color_scheme == "fire":
            # Red to yellow gradient
            r = 255
            g = int(255 * position)
            b = 0
            return (r, g, b)
        elif self.color_scheme == "ocean":
            # Deep blue to cyan gradient
            r = 0
            g = int(155 * position)
            b = 155 + int(100 * position)
            return (r, g, b)
        elif self.color_scheme == "forest":
            # Dark green to light green gradient
            r = int(50 * position)
            g = 100 + int(155 * position)
            b = int(50 * position)
            return (r, g, b)
        elif self.color_scheme == "neon":
            # Purple to pink to blue cycle
            if position < 0.33:
                # Purple to pink
                r = 200 + int(55 * (position * 3))
                g = 0
                b = 200 + int(55 * (position * 3))
            elif position < 0.66:
                # Pink to blue
                r = 255 - int(255 * ((position - 0.33) * 3))
                g = 0
                b = 255
            else:
                # Blue to purple
                r = int(200 * ((position - 0.66) * 3))
                g = 0
                b = 255
            return (r, g, b)
        elif self.color_scheme == "pastels":
            # Soft pastel colors
            r = 155 + int(100 * math.sin(position * 6.28))
            g = 155 + int(100 * math.sin((position * 6.28) + 2.09))
            b = 155 + int(100 * math.sin((position * 6.28) + 4.18))
            return (r, g, b)
        else:
            # Default to rainbow
            return self._hsv_to_rgb(position, 1.0, intensity)


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
                color = self._hsv_to_rgb(self.hue, 1.0, brightness)
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
            r, g, b = self._hsv_to_rgb(hue, sat, val)
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
            self.led_colors[i] = self._hsv_to_rgb(hue, 1.0, intensity)
        
        return self.get_colors()


class BassImpactVisualizer(BaseVisualizer):
    """Visualizer that emphasizes bass frequencies with powerful visual impact"""
    
    def __init__(self, num_leds: int = NUM_LEDS):
        super().__init__(num_leds)
        # Bass tracking
        self.bass_energy = 0.0
        self.bass_decay = 0.8
        self.bass_peak = 0.0
        self.peak_decay = 0.985  # Slower peak decay for better dynamic range
        
        # Dynamic range control
        self.min_threshold = 0.05  # Minimum threshold to avoid complete darkness
        self.dynamic_range = 0.95  # Maximum range between min and max response
        self.response_curve = 3  # Non-linear response curve (higher = more exponential)
        
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
            
            # Track peak bass for normalization (with a minimum floor to prevent division by zero)
            floor_value = 0.01  # Minimum floor value
            if self.bass_energy > self.bass_peak:
                self.bass_peak = max(self.bass_energy, floor_value)
            else:
                self.bass_peak = max(self.bass_peak * self.peak_decay + self.bass_energy * (1 - self.peak_decay), floor_value)
        else:
            # Decay if no data
            self.bass_energy *= self.bass_decay
            self.mid_energy *= self.mid_decay
            self.high_energy *= self.high_decay
        
        # Apply dynamic range and response curve for better bass expressiveness
        if self.bass_peak > 0:
            # Initial normalization
            raw_normalized = min(1.0, self.bass_energy / self.bass_peak)
            
            # Apply response curve (power function) for non-linear response
            # This makes small changes more visible while preventing constant peaking
            curved_response = pow(raw_normalized, 1.0 / self.response_curve)
            
            # Apply dynamic range adjustment
            normalized_bass = self.min_threshold + (curved_response * self.dynamic_range)
        else:
            normalized_bass = self.min_threshold
        
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
            pulse_factor = 1.0 - abs(pos - 0.5) * 2  # Center position factor (1.0 at center, 0.0 at edges)
            
            # Use normalized_bass to dynamically adjust the pulse intensity
            pulse_intensity = normalized_bass * pulse_factor
            
            # Create color based on frequency balance - boost mid and high colors
            # to maintain color balance even at lower bass levels
            r = int(self.bass_color[0] * pulse_intensity)
            g = int(self.mid_color[1] * self.mid_energy / max(self.bass_peak * 0.5, 0.01) * pulse_intensity)
            b = int(self.high_color[2] * self.high_energy / max(self.bass_peak * 0.5, 0.01) * pulse_intensity)
            
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
                color = self._hsv_to_rgb(hue, 1.0, self.brightness)
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


class CenterGradientVisualizer(BaseVisualizer):
    """Visualizer that creates an expanding pattern from center based on audio amplitude with cooling effect"""
    
    def __init__(self, num_leds: int = NUM_LEDS):
        super().__init__(num_leds)
        # Energy trackers for audio amplitude
        self.audio_energy = 0.0
        # Decay rates for smoothing
        self.energy_decay = 0.8
        # Previous frame energy for smoothing
        self.prev_energy = 0.0
        # Smoothing factor for changes between frames (0-1)
        self.transition_smoothing = 0.7
        # Minimum expansion (always shows a small central dot even with no audio)
        self.min_expansion = 0.05
        
        # Dynamic range adaptation
        self.amplitude_history = []
        self.history_max_size = 300  # Store last ~5 seconds at 60fps
        self.adaptation_rate = 0.02  # How quickly to adapt to new levels (0-1)
        self.amplitude_floor = 0.05  # Minimum amplitude threshold
        self.amplitude_ceiling = 1.0  # Maximum amplitude threshold
        self.dynamic_range = 0.7     # Amount of dynamic range scaling to apply (0-1)
        self.response_curve = 2.0    # Exponential curve for response (higher = more contrast)
        self.average_amplitude = 0.2 # Starting average estimate
        
        # Amplitude emphasis parameters
        self.emphasis_enabled = True    # Whether to enable dynamic emphasis
        self.emphasis_threshold = 0.15  # Range around average where emphasis applies (±percentage)
        self.emphasis_strength = 0.2    # How much to reduce brightness at average (0.0-1.0)
        self.emphasis_curve = 4.0       # How quickly emphasis falls off (higher = faster)
        
        # Cooling effect parameters
        self.use_color = True         # Whether to use colored cooling effect or stay white
        self.color_cooling = True     # Enable/disable cooling effect
        self.steady_state_counter = 0 # Counter for how long amplitude has been steady
        self.steady_state_threshold = 60 # Frames to trigger cooling effect (1 second at 60fps)
        self.amplitude_variance = 0.1 # How much amplitude can vary to be considered "steady"
        self.steady_state_range = []  # Recent amplitude values for variance calculation
        self.range_history_size = 30  # How many frames to consider for variance check
        self.current_hue = 0.0        # Current base hue (0-1)
        self.cooling_speed = 0.001    # How fast to cycle colors when in cooling mode
        self.cooling_saturation = 0.5 # Color saturation during cooling effect
        
        # Radiating cooling effect parameters
        self.cooling_radius = 0.0        # Current radius of the cooling effect (0.0-1.0)
        self.cooling_radius_speed = 0.02 # How fast the cooling effect propagates outward
        self.cooling_radius_max = 1.0    # Maximum radius of cooling effect (1.0 = full strip)
        self.cooling_active_prev = False # Track if cooling was active on previous frame
        
        # Color palette when in cooling mode
        self.color_palette = [
            (0.0, 0.9, 0.9),  # Red (hue, saturation, value)
            (0.1, 0.9, 0.9),  # Orange
            (0.2, 0.9, 0.9),  # Yellow
            (0.33, 0.9, 0.9), # Green
            (0.5, 0.9, 0.9),  # Cyan
            (0.66, 0.9, 0.9), # Blue
            (0.83, 0.9, 0.9)  # Purple
        ]
    
    def update(self, audio_data=None, fft_data=None, beat_detected=False, frequency_bands=None) -> List[int]:
        """Update visualization with expanding pattern based on audio amplitude"""
        # Get parameters
        brightness = self.brightness
        sensitivity = self.sensitivity
        
        # Calculate current audio energy from different sources
        current_energy = 0.0
        
        if audio_data is not None and len(audio_data) > 0:
            # Calculate RMS (root mean square) of audio data
            audio_data = np.array(audio_data)
            if audio_data.ndim > 1:  # If stereo, convert to mono
                audio_data = np.mean(audio_data, axis=1)
            rms = np.sqrt(np.mean(np.square(audio_data)))
            current_energy = rms * 3.0 * sensitivity  # Scale up since RMS values are typically small
        elif frequency_bands is not None and len(frequency_bands) > 0:
            # Use overall energy from frequency bands
            # Weight bass a bit more heavily
            bass_weight = 1.5
            mid_weight = 1.0
            high_weight = 0.7
            
            # Calculate weighted average based on frequency bands
            if len(frequency_bands) >= 7:
                bass = np.mean(frequency_bands[:2]) * bass_weight
                mid = np.mean(frequency_bands[2:5]) * mid_weight
                high = np.mean(frequency_bands[5:]) * high_weight
                current_energy = (bass + mid + high) / (bass_weight + mid_weight + high_weight)
            else:
                current_energy = np.mean(frequency_bands)
            
            current_energy *= sensitivity
        
        # Update running average with decay
        self.audio_energy = max(current_energy, self.audio_energy * self.energy_decay)
        
        # Apply smoothing to energy changes
        smoothed_energy = (self.prev_energy * self.transition_smoothing + 
                         self.audio_energy * (1 - self.transition_smoothing))
        self.prev_energy = smoothed_energy
        
        # Update amplitude history for dynamic range adaptation
        self.amplitude_history.append(smoothed_energy)
        if len(self.amplitude_history) > self.history_max_size:
            self.amplitude_history.pop(0)  # Remove oldest value
        
        # Calculate dynamic average amplitude (with protection against empty list)
        if self.amplitude_history:
            # Calculate current average
            current_avg = np.mean(self.amplitude_history)
            # Update running average with adaptation rate
            self.average_amplitude = (self.average_amplitude * (1 - self.adaptation_rate) + 
                                    current_avg * self.adaptation_rate)
        
        # Apply adaptive scaling to current energy
        if self.average_amplitude > 0:
            # Calculate normalized energy relative to the average
            # Use dynamic_range parameter to control how much adaptation affects the scaling
            normalized_energy = smoothed_energy / (self.average_amplitude * 2)  # Scale factor of 2 helps ensure good dynamic range
            
            # Apply blending between raw and normalized energy based on dynamic_range parameter
            adapted_energy = (smoothed_energy * (1 - self.dynamic_range) + 
                           normalized_energy * self.dynamic_range)
            
            # Apply non-linear response curve for better visual contrast
            # This makes quiet parts darker and loud parts brighter
            curved_energy = min(1.0, pow(adapted_energy, 1.0 / self.response_curve))
            
            # Ensure that very quiet sounds still show minimal expansion
            # by blending with minimum expansion threshold
            expansion_factor = max(self.min_expansion, curved_energy)
        else:
            # Fallback if average is zero
            expansion_factor = max(self.min_expansion, smoothed_energy)
        
        # Calculate dynamic amplitude emphasis factor
        emphasis_factor = 1.0  # Default to full brightness (no emphasis)
        if self.emphasis_enabled and self.average_amplitude > 0.05:  # Only apply when we have sufficient audio data
            # Calculate how close current amplitude is to the average (normalized 0.0-1.0)
            # 0.0 = at exactly average, 1.0 = far from average
            amplitude_diff = abs(smoothed_energy - self.average_amplitude) / max(0.05, self.average_amplitude)
            
            # Calculate the threshold band as a ratio
            threshold_band = self.emphasis_threshold
            
            # If current amplitude is within threshold band around average, apply emphasis
            if amplitude_diff < threshold_band:
                # Calculate how close to center of band (0.0 = at average, 1.0 = at edge of band)
                # This creates a bell curve where the center gets maximum emphasis
                relative_position = amplitude_diff / threshold_band
                
                # Apply emphasis based on position within threshold band using a curve
                # At exactly average, we apply maximum emphasis (reducing brightness)
                # Near the edges of the band, emphasis decreases
                band_position = pow(1.0 - relative_position, self.emphasis_curve)
                
                # Calculate emphasis factor (reduces brightness when close to average)
                # 1.0 = no emphasis, down to (1.0 - emphasis_strength) at the center
                emphasis_factor = 1.0 - (self.emphasis_strength * band_position)
        
        # Update steady state detection (for cooling effect)
        self.steady_state_range.append(smoothed_energy)
        if len(self.steady_state_range) > self.range_history_size:
            self.steady_state_range.pop(0)
        
        # Calculate if we're in a steady state (amplitude doesn't vary much)
        in_steady_state = False
        if len(self.steady_state_range) >= 10:  # Need enough samples
            # Calculate variance in recent amplitude
            amplitude_min = min(self.steady_state_range)
            amplitude_max = max(self.steady_state_range)
            variance = amplitude_max - amplitude_min
            
            # Check if variance is below threshold and amplitude is not too low
            if variance < self.amplitude_variance and np.mean(self.steady_state_range) > 0.1:
                in_steady_state = True
        
        # Update steady state counter
        if in_steady_state:
            self.steady_state_counter += 1
            if self.steady_state_counter > 300:  # Cap at 5 seconds
                self.steady_state_counter = 300
        else:
            # Reset counter slowly to avoid flickering
            self.steady_state_counter = max(0, self.steady_state_counter - 2)
        
        # Determine if cooling effect is active
        cooling_active = self.color_cooling and self.use_color and self.steady_state_counter >= self.steady_state_threshold
        
        # Handle radiating cooling effect
        if cooling_active:
            # When first activating cooling, reset radius to start from center
            if not self.cooling_active_prev:
                self.cooling_radius = 0.0
            
            # Gradually increase cooling radius over time to create the radiating effect
            # The speed increases with the steady state counter to make it more dynamic
            radius_speed = self.cooling_radius_speed * (1.0 + min(1.0, (self.steady_state_counter - self.steady_state_threshold) / 120))
            self.cooling_radius = min(self.cooling_radius_max, self.cooling_radius + radius_speed)
            
            # Update color cycle
            cooling_speed = self.cooling_speed * (1.0 + min(1.0, (self.steady_state_counter - self.steady_state_threshold) / 60))
            self.current_hue = (self.current_hue + cooling_speed) % 1.0
        else:
            # Gradually decrease radius when cooling effect is no longer active
            # This creates a smooth transition back to white
            self.cooling_radius = max(0.0, self.cooling_radius - self.cooling_radius_speed * 2)
        
        # Save cooling state for next frame
        self.cooling_active_prev = cooling_active
        
        # Calculate the center point
        center = self.num_leds // 2
        
        # Max distance from center to edge
        max_distance = self.num_leds // 2
        
        # Calculate expansion threshold in LED units
        expansion_threshold = int(max_distance * expansion_factor)
        
        # Fill the LED strip with colors
        for i in range(self.num_leds):
            # Calculate distance from center in LED units
            distance_from_center = abs(i - center)
            
            # If beyond the expansion threshold, set to black
            if distance_from_center > expansion_threshold:
                self.led_colors[i] = (0, 0, 0)  # Black
            else:
                # Calculate relative position within the expanded area (0.0-1.0)
                # This creates a gradient from center to edge of the expanded area
                relative_pos = distance_from_center / max(1, expansion_threshold)
                
                # Calculate brightness based on position (center is brighter, edges are darker)
                # Use the original smoothed_energy for brightness to maintain dynamics
                center_brightness = min(1.0, smoothed_energy * 1.2)  # Boost center brightness a bit
                
                # Make the gradient more pronounced by using non-linear falloff
                # This creates more contrast between center and edges
                edge_falloff = pow(relative_pos, 0.7)  # <1.0 for slower initial falloff
                edge_brightness = center_brightness * (1.0 - edge_falloff)
                
                # Apply dynamic amplitude emphasis, followed by global brightness
                final_brightness = min(1.0, edge_brightness * emphasis_factor * brightness)
                
                # Calculate normalized distance for cooling effect (0.0-1.0)
                # This is used to determine if a pixel is within the cooling radius
                normalized_distance = distance_from_center / max_distance
                
                # Apply cooling effect only to LEDs within the current cooling radius
                if self.cooling_radius > 0 and normalized_distance <= self.cooling_radius:
                    # Calculate how far into the cooling wave this pixel is
                    # 0.0 = at the edge of the wave, 1.0 = at the center or well inside the wave
                    wave_position = min(1.0, (self.cooling_radius - normalized_distance) / self.cooling_radius)
                    
                    # Increase saturation near the edge of the cooling wave for a wave-like effect
                    # This creates a visual wave front at the edge of the propagation
                    wave_factor = 1.0 - pow(1.0 - wave_position, 2)  # Non-linear falloff from edge
                    
                    # Calculate color saturation (stronger at center, fading to edges)
                    position_saturation = max(0.0, self.cooling_saturation * (1.0 - relative_pos) * wave_factor)
                    
                    # Calculate cooling effect strength based on how long we've been in steady state
                    cooling_strength = min(1.0, (self.steady_state_counter - self.steady_state_threshold) / 60)
                    
                    # Blend between white and color based on cooling strength
                    effective_saturation = position_saturation * cooling_strength
                    
                    # Get the RGB color
                    r, g, b = self._hsv_to_rgb(self.current_hue, effective_saturation, final_brightness)
                    self.led_colors[i] = (r, g, b)
                else:
                    # Set white color with calculated brightness (original behavior)
                    white_value = int(255 * final_brightness)
                    self.led_colors[i] = (white_value, white_value, white_value)
        
        return self.get_colors()
    
    def set_param(self, param, value):
        """Set visualizer parameters"""
        super().set_param(param, value)
        
        if param == "transition_smoothing":
            try:
                value = float(value)
                self.transition_smoothing = max(0.0, min(0.9, value))
            except ValueError:
                pass
        elif param == "min_expansion":
            try:
                value = float(value)
                self.min_expansion = max(0.0, min(0.5, value))
            except ValueError:
                pass
        elif param == "energy_decay":
            try:
                value = float(value)
                self.energy_decay = max(0.5, min(0.95, value))
            except ValueError:
                pass
        elif param == "dynamic_range":
            try:
                value = float(value)
                self.dynamic_range = max(0.0, min(1.0, value))
            except ValueError:
                pass
        elif param == "response_curve":
            try:
                value = float(value)
                self.response_curve = max(0.5, min(5.0, value))
            except ValueError:
                pass
        elif param == "adaptation_rate":
            try:
                value = float(value)
                self.adaptation_rate = max(0.001, min(0.1, value))
            except ValueError:
                pass
        elif param == "use_color":
            self.use_color = bool(value)
        elif param == "color_cooling":
            self.color_cooling = bool(value)
        elif param == "steady_state_threshold":
            try:
                value = int(value)
                self.steady_state_threshold = max(15, min(300, value))
            except ValueError:
                pass
        elif param == "amplitude_variance":
            try:
                value = float(value)
                self.amplitude_variance = max(0.01, min(0.5, value))
            except ValueError:
                pass
        elif param == "cooling_speed":
            try:
                value = float(value)
                self.cooling_speed = max(0.001, min(0.05, value))
            except ValueError:
                pass
        elif param == "cooling_saturation":
            try:
                value = float(value)
                self.cooling_saturation = max(0.0, min(1.0, value))
            except ValueError:
                pass
        elif param == "cooling_radius_speed":
            try:
                value = float(value)
                self.cooling_radius_speed = max(0.001, min(0.1, value))
            except ValueError:
                pass
        # New parameters for dynamic amplitude emphasis
        elif param == "emphasis_enabled":
            self.emphasis_enabled = bool(value)
        elif param == "emphasis_threshold":
            try:
                value = float(value)
                self.emphasis_threshold = max(0.01, min(0.5, value))
            except ValueError:
                pass
        elif param == "emphasis_strength":
            try:
                value = float(value)
                self.emphasis_strength = max(0.1, min(0.9, value))
            except ValueError:
                pass
        elif param == "emphasis_curve":
            try:
                value = float(value)
                self.emphasis_curve = max(0.5, min(5.0, value))
            except ValueError:
                pass


# Register available visualizations
VISUALIZATIONS = {
    'beat_pulse': BeatPulseVisualizer,
    'spectrum': SpectrumVisualizer,
    'energy_beat': EnergyBeatVisualizer,
    'bass_impact': BassImpactVisualizer,
    'frequency_bars': FrequencyBarsVisualizer,
    'vu_meter': VUMeterVisualizer,
    'center_gradient': CenterGradientVisualizer,
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
    elif vis_type == "center_gradient":
        return CenterGradientVisualizer(num_leds)
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