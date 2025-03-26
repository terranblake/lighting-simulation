# Testing Guidelines

## Python Script Testing
- Unit tests for animation generation functions
- Serial communication tests with timeout (maximum 5 seconds)
- Run tests with `pytest -v tests/`

## Arduino Testing
- Serial communication validation tests
- LED control verification
- Performance testing (frame rate measurement)
- All tests should exit after completion or timeout (30 seconds max)

## Integration Testing
- End-to-end tests with Python sending data to Arduino
- Visual confirmation of LED animations
- Frame rate measurement tests
- All tests should include proper error handling and exit conditions 