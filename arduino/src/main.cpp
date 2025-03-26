#include <Arduino.h>
#include <FastLED.h>

// LED strip configuration
#define LED_PIN     6      // Pin connected to the LED strip
#define NUM_LEDS    60     // Number of LEDs in the strip
#define LED_TYPE    WS2812B
#define COLOR_ORDER GRB

// Serial communication configuration
#define BAUD_RATE   115200
#define HEADER_MARKER 0xAA

// Buffer for incoming data (3 bytes per LED for RGB values)
uint8_t serialBuffer[NUM_LEDS * 3];
CRGB leds[NUM_LEDS];

void setup() {
  // Initialize serial communication
  Serial.begin(BAUD_RATE);
  
  // Initialize FastLED
  FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS)
    .setCorrection(TypicalLEDStrip);
  FastLED.setBrightness(50);  // Set initial brightness
  
  // Set all LEDs to off at startup
  FastLED.clear();
  FastLED.show();
  
  // Send ready signal
  Serial.println("READY");
}

void loop() {
  // Check if data is available
  if (Serial.available() > 0) {
    // Check for header marker
    if (Serial.read() == HEADER_MARKER) {
      // Read the number of LEDs to update
      uint8_t numToUpdate = Serial.read();
      
      if (numToUpdate > 0 && numToUpdate <= NUM_LEDS) {
        // Read RGB data for each LED
        uint16_t bytesToRead = numToUpdate * 3;
        uint16_t bytesRead = 0;
        
        // Wait for all data to arrive with timeout
        unsigned long startTime = millis();
        while (bytesRead < bytesToRead && (millis() - startTime) < 1000) {
          if (Serial.available() > 0) {
            serialBuffer[bytesRead] = Serial.read();
            bytesRead++;
          }
        }
        
        // If we received all expected data
        if (bytesRead == bytesToRead) {
          // Update LED values
          for (uint8_t i = 0; i < numToUpdate; i++) {
            uint16_t dataIndex = i * 3;
            leds[i].r = serialBuffer[dataIndex];
            leds[i].g = serialBuffer[dataIndex + 1];
            leds[i].b = serialBuffer[dataIndex + 2];
          }
          
          // Display the updated LEDs
          FastLED.show();
          
          // Send acknowledgment
          Serial.write(0x01);
        } else {
          // Timeout occurred, send error
          Serial.write(0xFF);
        }
      }
    }
  }
} 