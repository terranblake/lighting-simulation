# LED Animation System

A system for controlling RGB LED strips using Python and Arduino via serial communication.

## Overview

This project allows you to generate LED animations in Python and send them to an Arduino that controls an RGB LED strip using the FastLED library. It now also supports audio visualization features to react to music or sound.

## Components

- **Python Animation Generator**: Creates RGB animations and sends them over serial
- **Arduino Controller**: Receives animation data and controls the LED strip
- **Communication Protocol**: Simple serial protocol for transferring RGB values
- **Audio Visualization**: Captures system audio and generates visualizations on the LED strip

## Hardware Requirements

- Arduino board (Uno, Nano, Mega, etc.)
- WS2812B RGB LED strip (60 LEDs)
- 5V power supply appropriate for powering the LED strip
- USB cable for connecting Arduino to computer
- For audio visualization: A virtual audio device like BlackHole (macOS) or VB-Cable (Windows)

## Software Requirements

- Python 3.6+
- PySerial library
- PlatformIO for Arduino development
- FastLED library
- For audio visualization: numpy, pyaudio, librosa, sounddevice, matplotlib, scipy

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

### Audio Setup (macOS)

1. Install BlackHole for virtual audio routing:
   ```bash
   brew install --cask blackhole-2ch
   ```

2. Restart your computer to complete the installation
   
3. Configure system audio:
   - Open System Settings > Sound
   - Create a Multi-Output Device that includes both your speakers and BlackHole
   - Set the Multi-Output Device as your output device

## Available Animations

- **Rainbow**: Colorful rainbow pattern that moves along the strip
- **Color Wipe**: Single colored dot that moves back and forth
- **Pulse**: Entire strip pulsing with changing colors

## Audio Visualizations

The system now supports audio visualization that reacts to music or other sounds:

- **Beat Pulse**: Pulses the entire strip on detected beats
- **Spectrum**: Maps audio frequency bands to different sections of the LED strip
- **Energy Beat**: Combines beat detection with energy levels for dynamic visuals

### Running Audio Visualizations

1. List available audio devices:
   ```bash
   python -m python_scripts.audio_visualizer_demo --list-devices
   ```

2. Run the visualization demo (without LEDs):
   ```bash
   python -m python_scripts.audio_visualizer_demo --device BlackHole --vis beat_pulse
   ```

3. Connect to Arduino and display on LED strip:
   ```bash
   python -m python_scripts.audio_led_controller --device BlackHole --vis beat_pulse --port [YOUR_PORT]
   ```

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

## Adding Custom Audio Visualizations

To add a new audio visualization, edit `python_scripts/audio_visualizations.py`:

1. Create a new class that inherits from BaseVisualizer
2. Implement the update() method to process audio data and generate LED colors
3. Add it to the VISUALIZATIONS dictionary

## Protocol

The communication protocol uses a simple binary format:
- Header: 0xAA (1 byte)
- Number of LEDs to update (1 byte)
- RGB values (3 bytes per LED)

## License

MIT License 