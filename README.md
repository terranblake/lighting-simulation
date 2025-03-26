# LED Animation System

A system for controlling RGB LED strips using Python and Arduino via serial communication.

## Overview

This project allows you to generate LED animations in Python and send them to an Arduino that controls an RGB LED strip using the FastLED library.

## Components

- **Python Animation Generator**: Creates RGB animations and sends them over serial
- **Arduino Controller**: Receives animation data and controls the LED strip
- **Communication Protocol**: Simple serial protocol for transferring RGB values

## Hardware Requirements

- Arduino board (Uno, Nano, Mega, etc.)
- WS2812B RGB LED strip (60 LEDs)
- 5V power supply appropriate for powering the LED strip
- USB cable for connecting Arduino to computer

## Software Requirements

- Python 3.6+
- PySerial library
- PlatformIO for Arduino development
- FastLED library

## Setup Instructions

### Arduino Setup

1. Connect the LED strip to the Arduino:
   - Data pin to pin 6 (default, can be changed in code)
   - Connect power and ground appropriately

2. Upload the Arduino code:
   ```bash
   cd arduino
   pio run --target upload
   ```

### Python Setup

1. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run an animation:
   ```bash
   python -m python_scripts.animation_sender --port [YOUR_PORT] --animation rainbow
   ```

## Available Animations

- **Rainbow**: Colorful rainbow pattern that moves along the strip
- **Color Wipe**: Single colored dot that moves back and forth
- **Pulse**: Entire strip pulsing with changing colors

## Hardware Testing

For hardware integration testing, use:

```bash
python -m python_scripts.hardware_test --port [YOUR_PORT]
```

This will automatically run through all animations to verify your hardware setup.

See the [Hardware Testing Guide](HARDWARE_TESTING.md) for detailed information on setting up and testing the hardware.

## Adding Custom Animations

To add a new animation, edit `python_scripts/animation_sender.py`:

1. Add a new animation function
2. Add it to the `ANIMATIONS` dictionary

## Protocol

The communication protocol uses a simple binary format:
- Header: 0xAA (1 byte)
- Number of LEDs to update (1 byte)
- RGB values (3 bytes per LED)

## License

MIT License 