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
#define ACK_SUCCESS  0x01
#define ACK_ERROR    0xFF
#define DEBUG_PRINT  0x02  // Special command for debug printing

// Debug mode (set to 1 to enable verbose output)
#define DEBUG_MODE  1

// Buffer for incoming data (3 bytes per LED for RGB values)
uint8_t serialBuffer[NUM_LEDS * 3];
CRGB leds[NUM_LEDS];

// Statistics
unsigned long frameCount = 0;
unsigned long lastFrameTime = 0;
float frameRate = 0;

// Forward declarations
void processFrame(uint8_t numLeds);
void sendDebugMessage(const char *message);

void setup() {
  // Initialize serial communication
  Serial.begin(BAUD_RATE);
  
  // Wait for serial to be ready
  delay(100);
  
  // Clear serial buffers
  while(Serial.available()) Serial.read();
  
  // Initialize FastLED
  FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS)
    .setCorrection(TypicalLEDStrip);
  FastLED.setBrightness(50);  // Set initial brightness
  
  // Set all LEDs to off at startup
  FastLED.clear();
  FastLED.show();
  
  // Send ready signal using binary protocol
  Serial.write(DEBUG_PRINT);
  Serial.println("READY");
  
  if (DEBUG_MODE) {
    sendDebugMessage("LED Animation Controller initialized");
    char buffer[64];
    sprintf(buffer, "LEDs: %d, Pin: %d", NUM_LEDS, LED_PIN);
    sendDebugMessage(buffer);
  }
}

void loop() {
  // Process incoming data
  if (Serial.available() >= 2) {  // We need at least 2 bytes (header + length)
    uint8_t header = Serial.read();
    
    if (header == HEADER_MARKER) {
      uint8_t numLeds = Serial.read();
      processFrame(numLeds);
    } else {
      // Invalid header, flush buffer
      if (DEBUG_MODE) {
        char buffer[32];
        sprintf(buffer, "Bad header: 0x%02X", header);
        sendDebugMessage(buffer);
      }
      
      // Discard any remaining data
      while (Serial.available()) {
        Serial.read();
      }
      
      // Send error
      Serial.write(ACK_ERROR);
    }
  }
}

void processFrame(uint8_t numLeds) {
  if (numLeds == 0 || numLeds > NUM_LEDS) {
    if (DEBUG_MODE) {
      char buffer[32];
      sprintf(buffer, "Invalid LED count: %d", numLeds);
      sendDebugMessage(buffer);
    }
    Serial.write(ACK_ERROR);
    return;
  }
  
  // Calculate bytes to read (3 bytes per LED)
  uint16_t bytesToRead = numLeds * 3;
  uint16_t bytesRead = 0;
  bool timeout = false;
  
  // Wait for all data to arrive with timeout
  unsigned long startTime = millis();
  while (bytesRead < bytesToRead) {
    if (millis() - startTime > 1000) {
      // Timeout
      timeout = true;
      break;
    }
    
    if (Serial.available() > 0) {
      serialBuffer[bytesRead] = Serial.read();
      bytesRead++;
      startTime = millis(); // Reset timeout for each byte
    }
  }
  
  // If we received all expected data
  if (!timeout && bytesRead == bytesToRead) {
    // Update LED values
    for (uint8_t i = 0; i < numLeds; i++) {
      uint16_t dataIndex = i * 3;
      leds[i].r = serialBuffer[dataIndex];
      leds[i].g = serialBuffer[dataIndex + 1];
      leds[i].b = serialBuffer[dataIndex + 2];
    }
    
    // Display the updated LEDs
    unsigned long updateStart = millis();
    FastLED.show();
    unsigned long updateTime = millis() - updateStart;
    
    // Calculate frame rate
    unsigned long currentTime = millis();
    if (lastFrameTime > 0) {
      frameRate = 0.9 * frameRate + 0.1 * (1000.0 / (currentTime - lastFrameTime));
    }
    lastFrameTime = currentTime;
    frameCount++;
    
    // Send binary acknowledgment
    Serial.write(ACK_SUCCESS);
    
    // Print debug info every 30 frames
    if (DEBUG_MODE && (frameCount % 30 == 0)) {
      char buffer[64];
      sprintf(buffer, "Frame: %lu, FPS: %.1f, Update: %lums", 
              frameCount, frameRate, updateTime);
      sendDebugMessage(buffer);
    }
  } else {
    // Timeout or incorrect data length
    if (DEBUG_MODE) {
      char buffer[64];
      sprintf(buffer, "Data error. Expected: %d bytes, Got: %d bytes", 
              bytesToRead, bytesRead);
      sendDebugMessage(buffer);
    }
    Serial.write(ACK_ERROR);
  }
}

void sendDebugMessage(const char *message) {
  // This uses a separate channel (DEBUG_PRINT command) to send text debug info
  // Python can choose to display or ignore these
  Serial.write(DEBUG_PRINT);
  Serial.println(message);
} 