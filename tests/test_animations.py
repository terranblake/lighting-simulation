#!/usr/bin/env python3
import unittest
from python_scripts.animation_sender import rainbow_animation, color_wipe_animation, pulse_animation

class TestAnimations(unittest.TestCase):
    
    def test_rainbow_animation(self):
        """Test that rainbow animation generates the correct number of LED colors"""
        num_leds = 60
        step = 10
        colors = rainbow_animation(num_leds, step)
        
        # Check that we get the right number of LEDs
        self.assertEqual(len(colors), num_leds)
        
        # Check that the colors are in valid range
        for r, g, b in colors:
            self.assertTrue(0 <= r <= 255, f"Red value {r} out of range")
            self.assertTrue(0 <= g <= 255, f"Green value {g} out of range")
            self.assertTrue(0 <= b <= 255, f"Blue value {b} out of range")
    
    def test_color_wipe_animation(self):
        """Test that color wipe animation works correctly"""
        num_leds = 60
        step = 5
        colors = color_wipe_animation(num_leds, step)
        
        # Check that we get the right number of LEDs
        self.assertEqual(len(colors), num_leds)
        
        # Only one LED should be lit
        lit_leds = [i for i, (r, g, b) in enumerate(colors) if (r, g, b) != (0, 0, 0)]
        self.assertEqual(len(lit_leds), 1, "Color wipe should have exactly one lit LED")
    
    def test_pulse_animation(self):
        """Test that pulse animation generates uniform colors for all LEDs"""
        num_leds = 60
        step = 20
        colors = pulse_animation(num_leds, step)
        
        # Check that we get the right number of LEDs
        self.assertEqual(len(colors), num_leds)
        
        # All LEDs should have the same color
        first_color = colors[0]
        for color in colors:
            self.assertEqual(color, first_color, "All LEDs should have the same color in pulse animation")
        
        # Check that colors are in valid range
        r, g, b = first_color
        self.assertTrue(0 <= r <= 255, f"Red value {r} out of range")
        self.assertTrue(0 <= g <= 255, f"Green value {g} out of range")
        self.assertTrue(0 <= b <= 255, f"Blue value {b} out of range")

if __name__ == "__main__":
    unittest.main() 