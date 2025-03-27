#include <Arduino.h>
#include <FastLED.h>

// LED strip configuration
#define LED_PIN     6      // Pin connected to the LED strip
#define NUM_LEDS    60     // Number of LEDs in the strip
#define LED_TYPE    WS2812B
#define COLOR_ORDER GRB

// Color smoothing configuration
#define COLOR_SMOOTHING 0.7  // Blend factor (0.0-1.0): higher = faster transitions, lower = smoother
#define ENABLE_SMOOTHING 0   // Set to 1 to enable color smoothing, 0 to disable for performance

// Serial communication configuration
#define BAUD_RATE   921600 // Increased from 460800 to maximum supported rate
#define HEADER_FULL 0xAA    // Full frame update
#define HEADER_DELTA 0xBB   // Differential frame update
#define BUFFER_CHECK 0x42   // ASCII 'B' - Check buffer status command
#define DEBUG_PRINT  0x02    // Debug message marker
#define BUFFER_STATUS 'B'  // Command to query buffer status

// Buffer management
#define SERIAL_BUFFER_SIZE 256  // Increased from 128 to handle higher baud rate
#define MAX_FRAMES_QUEUE 2     // Maximum number of frames to queue

// Debug mode (set to 1 to enable FPS reporting but optimize other debug code)
#define DEBUG_MODE  1
#define MINIMAL_DEBUG 1  // Set to 1 for minimal debug (only FPS reporting, no verbose messages)
#define LOG_FPS_INTERVAL 30  // Send FPS info every N frames

// Direct memory access buffer for maximum speed
CRGB leds[NUM_LEDS];

// Performance optimization: Pre-allocate buffers instead of allocating during frame processing
uint8_t serialBuffer[NUM_LEDS * 3];
uint8_t deltaBuffer[NUM_LEDS * 4]; // For delta frames: index + RGB

// Frame rate control
#define MIN_FRAME_INTERVAL 0  // Removed minimum frame interval to maximize throughput
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

// Batch processing
#define BATCH_SIZE 10        // Process LEDs in batches for delta frames
#define BATCH_SHOW_INTERVAL 0  // Only call show() after this many batches (0 = only after all batches)

// Forward declarations
void processFrame(uint8_t numLeds);
void processDeltaFrame(uint8_t numChangedLeds);
void sendDebugMessage(const char *message);
void sendFpsReport(float fps);
void sendBufferStatus();
float getBufferUsage();
bool waitForData(uint16_t bytesNeeded, uint16_t timeoutMs);
void flushInputBuffer(uint16_t maxBytes);
inline void readSerialData(uint8_t* buffer, uint16_t bytesToRead);

#define ACK_SUCCESS 0x01    // Acknowledgment - success
#define ACK_ERROR 0xFF      // Acknowledgment - error

void setup() {
  // Initialize serial communication
  Serial.begin(BAUD_RATE);
  
  // Wait for serial to be ready
  delay(100);
  
  // Clear serial buffers
  while(Serial.available()) Serial.read();
  
  // Initialize FastLED with performance optimizations
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
  sendDebugMessage("READY");
  
  if (DEBUG_MODE) {
    sendDebugMessage("LED Animation Controller initialized");
    if (!MINIMAL_DEBUG) {
      char buffer[64];
      sprintf(buffer, "LEDs: %d, Pin: %d, Buffer: %d bytes", 
              NUM_LEDS, LED_PIN, SERIAL_BUFFER_SIZE);
      sendDebugMessage(buffer);
    }
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
        if (DEBUG_MODE && !MINIMAL_DEBUG) {
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
        if (DEBUG_MODE && !MINIMAL_DEBUG) {
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
      
      if (DEBUG_MODE && !MINIMAL_DEBUG) {
        char buffer[32];
        sprintf(buffer, "Buffer status: %d%% full", (bufferStatus * 100) / 255);
        sendDebugMessage(buffer);
      }
    }
    // Unknown command
    else {
      if (DEBUG_MODE && !MINIMAL_DEBUG) {
        char buffer[24];
        sprintf(buffer, "Bad header: 0x%02X", header);
        sendDebugMessage(buffer);
      }
      
      // Flush a few bytes to try to recover from bad state
      flushInputBuffer(10);  // Read up to 10 bytes to try to resync
      Serial.write(ACK_ERROR);
    }
  }
  
  // Remove delay for maximum performance
  // delay(1);
}

// Optimized method to read serial data in bulk for better performance
inline void readSerialData(uint8_t* buffer, uint16_t bytesToRead) {
  uint16_t bytesRead = 0;
  while (bytesRead < bytesToRead) {
    if (Serial.available()) {
      int bytesAvailable = min(Serial.available(), bytesToRead - bytesRead);
      while (bytesAvailable-- > 0) {
        buffer[bytesRead++] = Serial.read();
      }
    }
  }
}

// Helper to wait for a specific amount of data with timeout
bool waitForData(uint16_t bytesNeeded, uint16_t timeoutMs) {
  unsigned long startTime = millis();
  while (Serial.available() < bytesNeeded) {
    if (millis() - startTime > timeoutMs) {
      return false;  // Timeout
    }
    // No delay for maximum performance
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
    if (DEBUG_MODE && !MINIMAL_DEBUG) {
      char buffer[32];
      sprintf(buffer, "Invalid LED count: %d", numLeds);
      sendDebugMessage(buffer);
    }
    Serial.write(ACK_ERROR);
    return;
  }
  
  // Calculate bytes to read (3 bytes per LED)
  uint16_t bytesToRead = numLeds * 3;
  
  // Wait for all data to arrive with timeout
  if (!waitForData(bytesToRead, 200)) {
    if (DEBUG_MODE && !MINIMAL_DEBUG) {
      sendDebugMessage("Timeout waiting for frame data");
    }
    flushInputBuffer(Serial.available());
    Serial.write(ACK_ERROR);
    return;
  }
  
  // Use optimized bulk read for better performance
  readSerialData(serialBuffer, bytesToRead);
  
  // Process the received data directly without temporary array
  for (uint8_t i = 0; i < numLeds; i++) {
    uint16_t dataIndex = i * 3;
    
    #if ENABLE_SMOOTHING
      // Apply smoothing between old and new colors
      if (!firstFrame) {
        leds[i].r = previousColors[i].r + (COLOR_SMOOTHING * (serialBuffer[dataIndex] - previousColors[i].r));
        leds[i].g = previousColors[i].g + (COLOR_SMOOTHING * (serialBuffer[dataIndex + 1] - previousColors[i].g));
        leds[i].b = previousColors[i].b + (COLOR_SMOOTHING * (serialBuffer[dataIndex + 2] - previousColors[i].b));
        
        // Save current as previous for next frame
        previousColors[i].r = serialBuffer[dataIndex];
        previousColors[i].g = serialBuffer[dataIndex + 1];
        previousColors[i].b = serialBuffer[dataIndex + 2];
      } else {
        // For first frame, just use the new colors directly
        leds[i].r = serialBuffer[dataIndex];
        leds[i].g = serialBuffer[dataIndex + 1];
        leds[i].b = serialBuffer[dataIndex + 2];
        previousColors[i] = leds[i];
      }
    #else
      // Skip smoothing for maximum speed
      leds[i].r = serialBuffer[dataIndex];
      leds[i].g = serialBuffer[dataIndex + 1];
      leds[i].b = serialBuffer[dataIndex + 2];
    #endif
  }
  
  if (firstFrame) {
    firstFrame = false;
  }
  
  // Display the updated LEDs
  unsigned long updateStart = millis();
  
  // Performance optimization: Disable interpolation and dithering
  FastLED.setDither(0);
  FastLED.setMaxRefreshRate(0, false);
  FastLED.show();
  
  unsigned long updateTime = millis() - updateStart;
  
  // Calculate frame rate - always track for FPS reporting
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
  
  // Track total processing time
  totalProcessingTime += (millis() - startProcessing);
  
  // Send binary acknowledgment
  Serial.write(ACK_SUCCESS);
  
  // Print debug info periodically
  if (DEBUG_MODE && (frameCount % LOG_FPS_INTERVAL == 0)) {
    // Always send the FPS info - this is important for the Python client
    sendFpsReport(frameRate);
    
    // Only send extended info if minimal debug is off
    if (!MINIMAL_DEBUG) {
      char buffer[64];
      // Calculate average processing time
      unsigned long avgProcessTime = totalProcessingTime / (frameCount > 0 ? frameCount : 1);
      
      sprintf(buffer, "Frame: %lu, FPS: %d.%d, Update: %lums, Dropped: %lu, AvgProc: %lums", 
              frameCount, (int)frameRate, (int)((frameRate - (int)frameRate) * 10), 
              updateTime, droppedFrames, avgProcessTime);
      sendDebugMessage(buffer);
    }
  }
}

void processDeltaFrame(uint8_t numChangedLeds) {
  // Performance optimization: Remove frame rate limiting
  // Skip frame if we're processing frames too quickly
  // if (lastFrameProcessed > 0 && frameTime - lastFrameProcessed < MIN_FRAME_INTERVAL) {
  //   // Drop this frame to maintain frame rate
  //   droppedFrames++;
  //   flushInputBuffer(numChangedLeds * 4);
  //   Serial.write(ACK_SUCCESS);
  //   return;
  // }
  
  // Calculate bytes to read (index + RGB = 4 bytes per changed LED)
  uint16_t bytesToRead = numChangedLeds * 4;
  
  // Check if frame size is reasonable
  if (numChangedLeds > NUM_LEDS) {
    if (DEBUG_MODE && !MINIMAL_DEBUG) {
      char buffer[48];
      sprintf(buffer, "Invalid delta frame size: %d LEDs", numChangedLeds);
      sendDebugMessage(buffer);
    }
    flushInputBuffer(bytesToRead);  // Discard data
    Serial.write(ACK_ERROR);
    return;
  }
  
  // Wait for all the data to arrive
  if (!waitForData(bytesToRead, 200)) {
    // Timeout waiting for data
    if (DEBUG_MODE && !MINIMAL_DEBUG) {
      sendDebugMessage("Timeout waiting for delta frame data");
    }
    flushInputBuffer(Serial.available());  // Discard partial data
    Serial.write(ACK_ERROR);
    return;
  }
  
  // Performance optimization: Read all data in one batch
  readSerialData(deltaBuffer, bytesToRead);
  
  // Now process the data in batches for better performance
  bool allDataValid = true;
  bool shouldUpdate = false;
  
  for (uint8_t i = 0; i < numChangedLeds; i++) {
    // Get data from the buffer
    uint8_t ledIndex = deltaBuffer[i*4];
    uint8_t r = deltaBuffer[i*4 + 1];
    uint8_t g = deltaBuffer[i*4 + 2];
    uint8_t b = deltaBuffer[i*4 + 3];
    
    // Sanity check the LED index
    if (ledIndex < NUM_LEDS) {
      leds[ledIndex].r = r;
      leds[ledIndex].g = g;
      leds[ledIndex].b = b;
      shouldUpdate = true;
    } else {
      allDataValid = false;
    }
    
    // Update in batches for large delta frames
    if (BATCH_SHOW_INTERVAL > 0 && i > 0 && i % BATCH_SIZE == 0) {
      FastLED.show();
    }
  }
  
  // Update the LED strip only if something changed
  if (shouldUpdate) {
    // Performance optimization: Disable interpolation and dithering
    FastLED.setDither(0);
    FastLED.setMaxRefreshRate(0, false);
    FastLED.show();
  }
  
  lastFrameProcessed = millis();
  
  // Calculate frame rate - always track for FPS reporting
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
  
  // Send binary acknowledgment
  Serial.write(allDataValid ? ACK_SUCCESS : ACK_ERROR);
  
  // Send debug info periodically
  if (DEBUG_MODE && (frameCount % LOG_FPS_INTERVAL == 0)) {
    // Always send the FPS info - this is important for the Python client
    sendFpsReport(frameRate);
    
    // Only send extended info if minimal debug is off
    if (!MINIMAL_DEBUG) {
      char buffer[64];
      // Calculate average processing time
      unsigned long avgProcessTime = totalProcessingTime / (frameCount > 0 ? frameCount : 1);
      
      sprintf(buffer, "Frame: %lu, FPS: %d.%d, Dropped: %lu, AvgProc: %lums", 
              frameCount, (int)frameRate, (int)((frameRate - (int)frameRate) * 10), 
              droppedFrames, avgProcessTime);
      sendDebugMessage(buffer);
    }
  }
}

// Send FPS information in a standardized format for the Python client
void sendFpsReport(float fps) {
  int fps_int = (int)fps;
  int fps_dec = (int)((fps - fps_int) * 10);
  
  char fpsbuffer[20];
  sprintf(fpsbuffer, "FPS: %d.%d", fps_int, fps_dec);
  sendDebugMessage(fpsbuffer);
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
  
  if (DEBUG_MODE && !MINIMAL_DEBUG) {
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