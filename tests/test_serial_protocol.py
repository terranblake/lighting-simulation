#!/usr/bin/env python3
import unittest
import io
from unittest.mock import patch, MagicMock
from python_scripts.animation_sender import LEDAnimationSender

class TestSerialProtocol(unittest.TestCase):
    
    @patch('serial.Serial')
    def test_connection(self, mock_serial):
        # Setup mock
        mock_instance = MagicMock()
        mock_serial.return_value = mock_instance
        
        # Simulate READY response
        mock_instance.in_waiting = True
        mock_instance.readline.return_value = b'READY\n'
        
        # Create sender and connect
        sender = LEDAnimationSender(port='/dev/test', baud_rate=115200)
        result = sender.connect()
        
        # Check that connection was successful
        self.assertTrue(result)
        mock_serial.assert_called_with('/dev/test', 115200, timeout=2)
    
    @patch('serial.Serial')
    def test_send_frame(self, mock_serial):
        # Setup mock
        mock_instance = MagicMock()
        mock_serial.return_value = mock_instance
        mock_instance.is_open = True
        
        # Simulate successful acknowledgment
        mock_instance.in_waiting = True
        mock_instance.read.return_value = bytes([0x01])
        
        # Test data: 3 LEDs with RGB values
        test_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255)     # Blue
        ]
        
        # Create sender (skip connection)
        sender = LEDAnimationSender(port='/dev/test', baud_rate=115200)
        sender.serial = mock_instance
        
        # Send frame
        result = sender.send_frame(test_colors)
        
        # Check results
        self.assertTrue(result)
        
        # Verify correct protocol was followed
        expected_data = bytearray([
            0xAA,       # Header marker
            3,          # 3 LEDs to update
            255, 0, 0,  # LED 1: Red
            0, 255, 0,  # LED 2: Green
            0, 0, 255   # LED 3: Blue
        ])
        
        # Check that write was called with correct data
        call_args = mock_instance.write.call_args[0][0]
        self.assertEqual(len(call_args), len(expected_data))
        for i in range(len(expected_data)):
            self.assertEqual(call_args[i], expected_data[i])
    
    @patch('serial.Serial')
    def test_error_handling(self, mock_serial):
        # Setup mock
        mock_instance = MagicMock()
        mock_serial.return_value = mock_instance
        mock_instance.is_open = True
        
        # Simulate error acknowledgment
        mock_instance.in_waiting = True
        mock_instance.read.return_value = bytes([0xFF])
        
        # Test with a single color
        test_colors = [(255, 255, 255)]
        
        # Create sender (skip connection)
        sender = LEDAnimationSender(port='/dev/test', baud_rate=115200)
        sender.serial = mock_instance
        
        # Send frame - should get error response
        result = sender.send_frame(test_colors)
        
        # Check result is False due to error response
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main() 