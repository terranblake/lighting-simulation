# Terminal Operations Guidelines

## PlatformIO Commands
- Initialize project: `pio init --board uno`
- Build project: `pio run`
- Upload to Arduino: `pio run --target upload`
- Monitor serial output: `pio device monitor --baud 115200 --port /dev/ttyUSB0`

## Python Commands
- Run animation script: `python -m python_scripts.animation_sender --port /dev/ttyUSB0 --animation rainbow`
- Run tests: `pytest -v tests/`

## General Guidelines
- Always specify timeouts for any interactive process
- Use non-interactive mode for all commands when possible
- Redirect input/output as needed to avoid user interaction
- For serial monitoring, always use the `--exit-at pattern` option when available 