# TASK: LED Animation System

## Description
Create a system to send animation data from a Python script to an Arduino over serial communication. The Arduino will control an RGB LED strip using the FastLED library.

## Requirements
- Python script to generate RGB animation data
- Arduino code with FastLED library integration
- Serial communication between Python and Arduino
- Support for 60 RGB LEDs
- PlatformIO setup for Arduino code development and uploading

## Implementation Steps
1. Set up project structure
   - Create Python script directory
   - Set up PlatformIO project for Arduino

2. Implement Serial Communication Protocol
   - Define data format for LED animations
   - Implement serial communication in Python
   - Implement serial receiver in Arduino

3. Implement LED Control with FastLED
   - Initialize FastLED with correct LED type and pin
   - Process incoming animation data
   - Apply updates to LED strip

4. Create Test Animations in Python
   - Implement simple test patterns (e.g., rainbow, color wipe)
   - Add animation timing control
   - Test data transmission

## Testing
- Test serial communication between Python and Arduino
- Verify LED control with test patterns
- Measure and optimize performance

## Completion Criteria
- Python script can generate and send various animations
- Arduino correctly receives and displays animations on the LED strip
- System supports at least 30fps animation updates
- Documentation for using and extending the system 