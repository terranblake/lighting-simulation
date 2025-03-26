#!/usr/bin/env python3
"""
Hardware integration test for LED animation system.
This script runs a sequence of animations on the connected Arduino to verify
that hardware integration is working properly.
"""

import time
import argparse
from animation_sender import LEDAnimationSender, ANIMATIONS

def run_integration_test(port, duration=3, fps=30):
    """Run a sequence of animations to test hardware integration"""
    sender = LEDAnimationSender(port=port, baud_rate=115200, num_leds=60)
    
    # Test each animation pattern
    for name, animation_func in ANIMATIONS.items():
        print(f"\n{'='*50}")
        print(f"Testing animation: {name}")
        print(f"{'='*50}")
        
        sender.run_animation(animation_func, duration=duration, fps=fps)
        # Pause between animations
        time.sleep(1)
    
    print("\nHardware integration test complete!")

def main():
    parser = argparse.ArgumentParser(description='LED Hardware Integration Test')
    parser.add_argument('--port', required=True, help='Serial port connected to Arduino')
    parser.add_argument('--duration', type=int, default=3, help='Duration to test each animation (seconds)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for animations')
    
    args = parser.parse_args()
    
    print(f"Starting hardware integration test on port {args.port}")
    print("Make sure Arduino is connected with LED strip on pin 6")
    print("Press Ctrl+C to cancel or wait 3 seconds to continue...")
    
    try:
        time.sleep(3)
        run_integration_test(args.port, args.duration, args.fps)
    except KeyboardInterrupt:
        print("\nTest cancelled by user")
    except Exception as e:
        print(f"\nError during test: {e}")

if __name__ == "__main__":
    main() 