#include <Arduino.h>
#include <FastLED.h>

// LED strip configuration
#define LED_PIN     6      // Pin connected to the LED strip
#define NUM_LEDS    60     // Number of LEDs in the strip
#define LED_TYPE    WS2812B
#define COLOR_ORDER GRB

// Color smoothing configuration
#define COLOR_SMOOTHING 0.5  // Blend factor (0.0-1.0): higher = faster transitions, lower = smoother

// Serial communication configuration
#define BAUD_RATE   115200
#define HEADER_FULL 0xAA    // Full frame update
#define HEADER_DELTA 0xBB   // Differential frame update
#define BUFFER_CHECK 0x42   // ASCII 'B' - Check buffer status command
#define DEBUG_PRINT  0x02    // Debug message marker
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
#define MIN_FRAME_INTERVAL 8  // ~120fps = 8ms between frames
unsigned long lastFrameProcessed = 0;
unsigned long totalProcessingTime = 0;

// Statistics
unsigned long frameCount = 0;
unsigned long lastFrameTime = 0;
float frameRate = 0;
unsigned long droppedFrames = 0;
unsigned long bufferOverflows = 0;

// Previous color values for smoothing
CRGB previousColors[NUM_LEDS];
bool firstFrame = true;

// Forward declarations
void processFrame(uint8_t numLeds);
void processDeltaFrame(uint8_t numChangedLeds);
void sendDebugMessage(const char *message);
void sendBufferStatus();
float getBufferUsage();
bool waitForData(uint16_t bytesNeeded, uint16_t timeoutMs);
void flushInputBuffer(uint16_t maxBytes);

#define ACK_SUCCESS 0x01    // Acknowledgment - success
#define ACK_ERROR 0xFF      // Acknowledgment - error

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
  
  // Initialize smoothing arrays
  for (int i = 0; i < NUM_LEDS; i++) {
    previousColors[i] = CRGB::Black;
  }
  
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
  // Check if there's serial data available
  if (Serial.available() > 0) {
    uint8_t header = Serial.read();
    
    // Check for header bytes we understand
    if (header == HEADER_FULL) {
      // Read number of LEDs
      if (waitForData(1, 100)) {  // Wait for 1 byte with 100ms timeout
        uint8_t numLeds = Serial.read();
        processFrame(numLeds);
      } else {
        // Timeout waiting for LED count
        if (DEBUG_MODE) {
          sendDebugMessage("Timeout waiting for LED count");
        }
        Serial.write(ACK_ERROR);
      }
    }
    else if (header == HEADER_DELTA) {
      // Read number of changed LEDs
      if (waitForData(1, 100)) {  // Wait for 1 byte with 100ms timeout
        uint8_t numChangedLeds = Serial.read();
        processDeltaFrame(numChangedLeds);
      } else {
        // Timeout waiting for changed LED count
        if (DEBUG_MODE) {
          sendDebugMessage("Timeout waiting for changed LED count");
        }
        Serial.write(ACK_ERROR);
      }
    }
    // Buffer status check
    else if (header == BUFFER_CHECK) {
      // Respond with buffer status (0-255 representing 0-100%)
      uint8_t bufferStatus = (uint8_t)(getBufferUsage() * 255);
      Serial.write(bufferStatus);
      
      if (DEBUG_MODE) {
        char buffer[32];
        sprintf(buffer, "Buffer status: %d%% full", (bufferStatus * 100) / 255);
        sendDebugMessage(buffer);
      }
    }
    // Unknown command
    else {
      if (DEBUG_MODE) {
        char buffer[24];
        sprintf(buffer, "Bad header: 0x%02X", header);
        sendDebugMessage(buffer);
      }
      
      // Flush a few bytes to try to recover from bad state
      flushInputBuffer(10);  // Read up to 10 bytes to try to resync
      Serial.write(ACK_ERROR);
    }
  }
  
  // Very small delay to prevent hogging the CPU
  delayMicroseconds(100);
}

// Helper to wait for a specific amount of data with timeout
bool waitForData(uint16_t bytesNeeded, uint16_t timeoutMs) {
  unsigned long startTime = millis();
  while (Serial.available() < bytesNeeded) {
    if (millis() - startTime > timeoutMs) {
      return false;  // Timeout
    }
    delayMicroseconds(100);  // Short delay to avoid hogging CPU
  }
  return true;
}

// Helper to flush input buffer
void flushInputBuffer(uint16_t maxBytes) {
  uint16_t count = 0;
  while (Serial.available() > 0 && count < maxBytes) {
    Serial.read();
    count++;
  }
}

void processFrame(uint8_t numLeds) {
  unsigned long startProcessing = millis();
  
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
    // Create temporary array for new colors
    CRGB newColors[numLeds];
    
    // Read new colors into temporary array
    for (uint8_t i = 0; i < numLeds; i++) {
      uint16_t dataIndex = i * 3;
      newColors[i].r = serialBuffer[dataIndex];
      newColors[i].g = serialBuffer[dataIndex + 1];
      newColors[i].b = serialBuffer[dataIndex + 2];
    }
    
    // Apply smoothing between old and new colors
    if (!firstFrame) {
      for (uint8_t i = 0; i < numLeds; i++) {
        leds[i].r = previousColors[i].r + (COLOR_SMOOTHING * (newColors[i].r - previousColors[i].r));
        leds[i].g = previousColors[i].g + (COLOR_SMOOTHING * (newColors[i].g - previousColors[i].g));
        leds[i].b = previousColors[i].b + (COLOR_SMOOTHING * (newColors[i].b - previousColors[i].b));
        
        // Save current as previous for next frame
        previousColors[i] = newColors[i];
      }
    } else {
      // For first frame, just use the new colors directly
      for (uint8_t i = 0; i < numLeds; i++) {
        leds[i] = newColors[i];
        previousColors[i] = newColors[i];
      }
      firstFrame = false;
    }
    
    // Display the updated LEDs
    unsigned long updateStart = millis();
    
    // Set delay time to 0 for maximum speed
    FastLED.setMaxRefreshRate(0, true);
    FastLED.show();
    
    unsigned long updateTime = millis() - updateStart;
    
    // Calculate frame rate
    unsigned long currentTime = millis();
    if (lastFrameTime > 0) {
      float timeDiff = currentTime - lastFrameTime;
      if (timeDiff > 0) {
        // Valid time difference, calculate frameRate with smoothing
        frameRate = 0.7 * frameRate + 0.3 * (1000.0 / timeDiff); // More responsive smoothing
      }
      // Explicitly cap at 0 if something went wrong
      if (frameRate < 0 || isnan(frameRate)) {
        frameRate = 0.0;
      }
    } else {
      frameRate = 0.0;  // Initialize with 0 if it's the first frame
    }
    lastFrameTime = currentTime;
    frameCount++;
    
    // Track total processing time
    totalProcessingTime += (millis() - startProcessing);
    
    // Send binary acknowledgment
    Serial.write(ACK_SUCCESS);
    
    // Print debug info every 30 frames
    if (DEBUG_MODE && (frameCount % 30 == 0)) {
      char buffer[64];
      int fps_int = (int)frameRate;
      int fps_dec = (int)((frameRate - fps_int) * 10);
      
      // Calculate average processing time
      unsigned long avgProcessTime = totalProcessingTime / (frameCount > 0 ? frameCount : 1);
      
      sprintf(buffer, "Frame: %lu, FPS: %d.%d, Update: %lums, Dropped: %lu, AvgProc: %lums", 
              frameCount, fps_int, fps_dec, updateTime, droppedFrames, avgProcessTime);
      sendDebugMessage(buffer);
      
      // Send a separate clean FPS message to make parsing easier
      char fpsbuffer[20];
      sprintf(fpsbuffer, "FPS: %d.%d", fps_int, fps_dec);
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

void processDeltaFrame(uint8_t numChangedLeds) {
  // Skip frame if we're processing frames too quickly
  unsigned long frameTime = millis();
  if (lastFrameProcessed > 0 && frameTime - lastFrameProcessed < MIN_FRAME_INTERVAL) {
    // Drop this frame to maintain frame rate
    droppedFrames++;
    
    // Read and discard the expected data (4 bytes per changed LED)
    flushInputBuffer(numChangedLeds * 4);
    
    // Send acknowledgment
    Serial.write(ACK_SUCCESS);
    return;
  }
  
  // Calculate bytes to read (index + RGB = 4 bytes per changed LED)
  uint16_t bytesToRead = numChangedLeds * 4;
  
  // Check if frame size is reasonable
  if (numChangedLeds > NUM_LEDS) {
    if (DEBUG_MODE) {
      char buffer[48];
      sprintf(buffer, "Invalid delta frame size: %d LEDs", numChangedLeds);
      sendDebugMessage(buffer);
    }
    flushInputBuffer(bytesToRead);  // Discard data
    Serial.write(ACK_ERROR);
    return;
  }
  
  // Wait for all the data to arrive
  if (!waitForData(bytesToRead, 100)) {
    // Timeout waiting for data
    if (DEBUG_MODE) {
      sendDebugMessage("Timeout waiting for delta frame data");
    }
    flushInputBuffer(Serial.available());  // Discard partial data
    Serial.write(ACK_ERROR);
    return;
  }
  
  // Now read and process the data
  unsigned long updateStart = millis();
  bool allDataValid = true;
  
  // Read and process all changed LEDs
  for (uint8_t i = 0; i < numChangedLeds; i++) {
    // Read LED index and color
    uint8_t ledIndex = Serial.read();
    uint8_t r = Serial.read();
    uint8_t g = Serial.read();
    uint8_t b = Serial.read();
    
    // Sanity check the LED index
    if (ledIndex < NUM_LEDS) {
      leds[ledIndex].r = r;
      leds[ledIndex].g = g;
      leds[ledIndex].b = b;
    } else {
      allDataValid = false;
    }
  }
  
  // Update the LED strip
  FastLED.setMaxRefreshRate(0, true);
  FastLED.show();
  
  unsigned long updateTime = millis() - updateStart;
  lastFrameProcessed = millis();
  
  // Calculate frame rate
  unsigned long currentTime = millis();
  if (lastFrameTime > 0) {
    float timeDiff = currentTime - lastFrameTime;
    if (timeDiff > 0) {
      frameRate = 0.7 * frameRate + 0.3 * (1000.0 / timeDiff);
      if (frameRate < 0 || isnan(frameRate)) {
        frameRate = 0.0;
      }
    }
  } else {
    frameRate = 0.0;
  }
  lastFrameTime = currentTime;
  frameCount++;
  
  // Track processing time
  totalProcessingTime += (millis() - updateStart);
  
  // Send binary acknowledgment
  Serial.write(allDataValid ? ACK_SUCCESS : ACK_ERROR);
  
  // Send debug info periodically
  if (DEBUG_MODE && (frameCount % 30 == 0)) {
    char buffer[64];
    int fps_int = (int)frameRate;
    int fps_dec = (int)((frameRate - fps_int) * 10);
    
    // Calculate average processing time
    unsigned long avgProcessTime = totalProcessingTime / (frameCount > 0 ? frameCount : 1);
    
    sprintf(buffer, "Frame: %lu, FPS: %d.%d, Update: %lums, Dropped: %lu, AvgProc: %lums", 
            frameCount, fps_int, fps_dec, updateTime, droppedFrames, avgProcessTime);
    sendDebugMessage(buffer);
    
    // Send a separate clean FPS message to make parsing easier
    char fpsbuffer[20];
    sprintf(fpsbuffer, "FPS: %d.%d", fps_int, fps_dec);
    sendDebugMessage(fpsbuffer);
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
  float usage = (float)Serial.available() / SERIAL_BUFFER_SIZE;
  if (usage < 0.0) return 0.0;
  if (usage > 1.0) return 1.0;
  return usage;
} 