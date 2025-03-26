# Hardware Testing Guide

This guide provides instructions for testing the LED animation system with physical hardware.

## Hardware Setup

1. **Arduino Connection**:
   - Connect your Arduino board to your computer via USB

2. **LED Strip Connection**:
   - Connect the LED strip data pin to Arduino pin 6
   - Connect LED strip power and ground appropriately
   - Make sure to use a sufficient power supply for your LED strip

   ```
   Arduino                 LED Strip
   -------                 ---------
   Pin 6       --------→   Data In
   5V/GND      --------→   Power/GND (for small strips only)
   ```

   **Note**: For strips with more than a few LEDs, use an external 5V power supply.

## Upload Arduino Code

First, upload the code to your Arduino using PlatformIO:

```bash
cd arduino
pio run --target upload
```

## Determine Serial Port

Find your Arduino's serial port:

- **macOS**: `/dev/tty.usbserial*` or `/dev/tty.usbmodem*`
- **Linux**: `/dev/ttyUSB0` or `/dev/ttyACM0`
- **Windows**: `COM3` (or similar)

You can list available ports with:
```bash
python -c "import serial.tools.list_ports; print([p.device for p in serial.tools.list_ports.comports()])"
```

## Run Hardware Tests

Run the hardware testing script with your serial port:

```bash
python -m python_scripts.hardware_test --port /dev/your_port --duration 3 --fps 30
```

The test will run each animation for the specified duration.

## Troubleshooting

### No LED Response
- Verify correct pin connections
- Check power supply
- Ensure Arduino is properly programmed
- Confirm serial port selection

### Communication Errors
- Check serial port name
- Verify baud rate matches (115200)
- Restart Arduino and test script

### Performance Issues
- Reduce FPS if Arduino can't keep up
- Try with fewer LEDs if performance is poor 