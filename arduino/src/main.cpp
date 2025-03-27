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
#define BUFFER_STATUS 'B'  // Command to query buffer status

// Buffer management
#define SERIAL_BUFFER_SIZE 128  // Arduino Uno/Nano hardware serial buffer size
#define MAX_FRAMES_QUEUE 5     // Maximum number of frames to queue

// Debug mode (set to 1 to enable verbose output)
#define DEBUG_MODE  1

// Buffer for incoming data (3 bytes per LED for RGB values)
uint8_t serialBuffer[NUM_LEDS * 3];
CRGB leds[NUM_LEDS];

// Frame rate control
#define MIN_FRAME_INTERVAL 33  // 30fps = ~33ms between frames
unsigned long lastFrameProcessed = 0;

// Statistics
unsigned long frameCount = 0;
unsigned long lastFrameTime = 0;
float frameRate = 0;
unsigned long droppedFrames = 0;
unsigned long bufferOverflows = 0;

// Forward declarations
void processFrame(uint8_t numLeds);
void sendDebugMessage(const char *message);
void sendBufferStatus();
float getBufferUsage();

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
    sprintf(buffer, "LEDs: %d, Pin: %d, Buffer: %d bytes", 
            NUM_LEDS, LED_PIN, SERIAL_BUFFER_SIZE);
    sendDebugMessage(buffer);
  }
}

void loop() {
  // Process incoming data
  if (Serial.available() > 0) {
    uint8_t cmd = Serial.peek();
    
    // Check for buffer status request
    if (cmd == BUFFER_STATUS) {
      Serial.read(); // Consume the command byte
      sendBufferStatus();
      return;
    }
    
    // Check for animation frame
    if (cmd == HEADER_MARKER && Serial.available() >= 2) {
      Serial.read(); // Consume header
      uint8_t numLeds = Serial.read();
      
      // Check if we're processing frames too quickly
      unsigned long currentTime = millis();
      if (currentTime - lastFrameProcessed < MIN_FRAME_INTERVAL) {
        // Drop this frame to maintain frame rate
        // Discard the expected data
        uint16_t bytesToDiscard = numLeds * 3;
        for (uint16_t i = 0; i < bytesToDiscard && Serial.available(); i++) {
          Serial.read();
        }
        droppedFrames++;
        
        if (DEBUG_MODE && droppedFrames % 10 == 1) {
          char buffer[32];
          sprintf(buffer, "Dropped frame: %lu", droppedFrames);
          sendDebugMessage(buffer);
        }
        return;
      }
      
      // Process the frame
      processFrame(numLeds);
      lastFrameProcessed = currentTime;
    } else if (Serial.available() >= 2 && cmd != HEADER_MARKER) {
      // Invalid header, flush buffer
      Serial.read(); // Consume the invalid byte
      
      if (DEBUG_MODE) {
        char buffer[32];
        sprintf(buffer, "Bad header: 0x%02X", cmd);
        sendDebugMessage(buffer);
      }
      
      // Discard some data to avoid buffer filling up
      for (int i = 0; i < 10 && Serial.available(); i++) {
        Serial.read();
      }
      
      bufferOverflows++;
    }
  }
  
  // If buffer is getting full, send a warning
  if (Serial.available() > SERIAL_BUFFER_SIZE * 0.8) {
    if (DEBUG_MODE) {
      sendDebugMessage("WARNING: Serial buffer almost full");
    }
    
    // Discard some data to prevent overflow
    while (Serial.available() > SERIAL_BUFFER_SIZE / 2) {
      Serial.read();
    }
    
    bufferOverflows++;
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
    } else {
      frameRate = 0.0;  // Initialize with 0 if it's the first frame
    }
    lastFrameTime = currentTime;
    frameCount++;
    
    // Send binary acknowledgment
    Serial.write(ACK_SUCCESS);
    
    // Print debug info every 30 frames
    if (DEBUG_MODE && (frameCount % 30 == 0)) {
      char buffer[64];
      sprintf(buffer, "Frame: %lu, FPS: %.1f, Update: %lums, Dropped: %lu", 
              frameCount, frameRate, updateTime, droppedFrames);
      sendDebugMessage(buffer);
      
      // Send a separate clean FPS message to make parsing easier
      char fpsbuffer[20];
      sprintf(fpsbuffer, "FPS: %.1f", frameRate);
      sendDebugMessage(fpsbuffer);
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

void sendBufferStatus() {
  // Respond with a byte representing buffer fullness (0-255)
  uint8_t bufferStatus = getBufferUsage() * 255;
  Serial.write(bufferStatus);
  
  if (DEBUG_MODE) {
    char buffer[64];
    sprintf(buffer, "Buffer status: %d%% full", (bufferStatus * 100) / 255);
    sendDebugMessage(buffer);
  }
}

float getBufferUsage() {
  // Calculate buffer usage as a ratio (0.0 to 1.0)
  return constrain((float)Serial.available() / SERIAL_BUFFER_SIZE, 0.0, 1.0);
} 