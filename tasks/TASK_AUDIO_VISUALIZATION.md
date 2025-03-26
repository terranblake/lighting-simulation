# TASK: Audio Visualization for LED Strip

## Description
Add audio visualization capability to the LED animation system. The system will capture system audio (including from Chrome browser), process the audio data, and display visualizations on the 60 LED strip via the existing Arduino interface.

## Requirements
- Capture system audio on macOS without requiring a microphone
- Initially implement beat detection with modular design for other visualization styles
- Optimize for maximum refresh rate supported by WS2812B LEDs
- Support audio source selection with defaults for testing
- Include visualization parameter controls with defaults for testing
- Focus on macOS compatibility

## Implementation Steps
1. Set up audio capture system
   - Implement system audio capture for macOS
   - Create audio buffer and processing pipeline
   - Add audio source selection capability

2. Implement audio data analysis
   - Create FFT or similar frequency analysis
   - Implement beat detection algorithm
   - Design modular analysis system for future visualization types

3. Develop LED visualizations
   - Create beat-responsive visualization
   - Add parameter controls (sensitivity, colors, response time)
   - Ensure visualization is optimized for 60x1 LED format

4. Integrate with existing LED control system
   - Connect visualization output to the existing serial communication system
   - Ensure smooth performance at maximum possible frame rate
   - Add visualization selection to the command-line interface

## Testing
- Test audio capture from different sources (system audio, Chrome browser)
- Measure and optimize visualization frame rate
- Verify visualization responds correctly to different audio characteristics
- Test parameter adjustments for visualization customization

## Completion Criteria
- System can capture and visualize audio from Chrome browser
- Visualization runs at minimum 30fps
- Beat detection works reliably with various music genres
- System is modular and allows for easy addition of other visualization types 