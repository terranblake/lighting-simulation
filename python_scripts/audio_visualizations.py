class BaseVisualizer:
    """Base class for LED visualizers."""
    
    def __init__(self, num_leds=NUM_LEDS, brightness=1.0):
        """Initialize with the number of LEDs."""
        self.num_leds = num_leds
        self.brightness = brightness
        self.sensitivity = 1.0  # Overall sensitivity multiplier
        self.led_colors = [(0, 0, 0)] * num_leds
        self.bottom_up = True  # Default to bottom-up display (index 0 at bottom)
    
    def update(self, audio_data=None, fft_data=None, beat_detected=False, frequency_bands=None):
        """Update the LED colors based on the audio data."""
        return self.get_colors()
        
    def get_colors(self):
        """Get the current LED colors."""
        if hasattr(self, 'led_colors') and self.led_colors:
            # If colors are stored as (r,g,b) tuples
            if isinstance(self.led_colors[0], tuple):
                # Apply brightness and convert to flat list for compatibility
                flat_colors = []
                for r, g, b in self.led_colors:
                    r = int(r * self.brightness)
                    g = int(g * self.brightness)
                    b = int(b * self.brightness)
                    flat_colors.extend([r, g, b])
                return flat_colors
            else:
                # Already flat format, return as-is
                return self.led_colors
        return [0] * (self.num_leds * 3)
    
    # Add compatibility method for web_visualizer.py
    def get_led_colors(self):
        """Compatibility method for web_visualizer - returns list of (r,g,b) tuples."""
        if hasattr(self, 'led_colors') and self.led_colors:
            # If already in correct tuple format
            if isinstance(self.led_colors[0], tuple):
                return self.led_colors
            else:
                # Convert flat [r,g,b,r,g,b,...] to list of tuples [(r,g,b), ...]
                tuple_colors = []
                for i in range(0, len(self.led_colors), 3):
                    if i + 2 < len(self.led_colors):
                        r = self.led_colors[i]
                        g = self.led_colors[i+1]
                        b = self.led_colors[i+2]
                        tuple_colors.append((r, g, b))
                return tuple_colors
        # Empty case
        return [(0, 0, 0)] * self.num_leds 