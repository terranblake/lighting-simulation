# LED Animation System

A system for controlling RGB LED strips using Python and Arduino via serial communication, with advanced audio visualization capabilities.

## Overview

This project allows you to generate LED animations in Python and send them to an Arduino that controls an RGB LED strip using the FastLED library. It includes sophisticated audio visualization features that react to music or sound in real-time, with support for frequency analysis, beat detection, and dynamic visual effects.

## Components

- **Python Animation Generator**: Creates RGB animations and sends them over serial
- **Arduino Controller**: Receives animation data and controls the LED strip
- **Communication Protocol**: Optimized binary protocol for transferring RGB values
- **Audio Visualization**: Captures system audio and generates visualizations on the LED strip
- **Web Interface**: Real-time control and preview of visualizations

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

## Available Visualizations

1. **Spectrum** - Displays the frequency spectrum across the LED strip
2. **Beat Pulse** - Pulses the entire strip on detected beats
3. **Energy Beat** - Creates flowing patterns based on audio energy and beats
4. **Bass Impact** - Emphasizes bass frequencies with powerful visual impact
5. **Frequency Bars** - Divides the LED strip into frequency bands
6. **VU Meter** - Mimics a volume unit meter with color gradients
7. **Center Gradient** - Creates an expanding pattern from center with dynamic effects:
   - Amplitude-driven expansion
   - Dynamic range adaptation
   - Color cooling effect during steady audio
   - Frequency-selective emphasis
   - Auto-calibration to most prominent frequency band

## Web Interface

The project includes a web interface for real-time control and preview:

1. Start the web interface:
   ```
   python python_scripts/web_visualizer.py
   ```

2. Open a web browser and go to http://localhost:5050

3. Control options:
   - Select audio device (BlackHole 2ch for system audio)
   - Choose visualization type
   - Adjust brightness
   - Configure amplitude emphasis
   - Enable/disable frequency-selective mode
   - Monitor performance metrics (FPS, buffer usage)

## Performance Optimizations

The system includes several optimizations for high performance:

- Serial communication at 921600 baud rate
- Optimized binary protocol for LED updates
- Efficient buffer management
- Frame rate limiting and monitoring
- Dynamic range adaptation
- Frequency-selective processing

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

The communication protocol uses an optimized binary format:
- Header: 0xAA (1 byte)
- Number of LEDs to update (1 byte)
- RGB values (3 bytes per LED)
- Support for partial frame updates
- Delta frame compression for efficiency

## License

MIT License 