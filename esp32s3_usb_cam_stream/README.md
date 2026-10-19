# ESP32S3-CAM USB camera stream

## Contents
- usb_cam_stream/usb_cam_stream.ino  -> Arduino sketch (open this folder in Arduino IDE)
- viewer.py                          -> live viewer that runs on your computer

## 1. Upload the sketch
Requires the ESP32 board package (Boards Manager -> "esp32" by Espressif).
The camera driver (esp_camera.h) is included with it; no extra libraries needed.

Tools settings:
- Board: ESP32S3 Dev Module
- USB CDC On Boot: Enabled
- Flash Size: 8MB (64Mb)
- Partition Scheme: 8M with spiffs (3MB APP/1.5MB SPIFFS)
- PSRAM: OPI PSRAM
- Upload Mode: UART0 / Hardware CDC
- USB Mode: Hardware CDC and JTAG
- Core Debug Level: None

Upload, then press RESET. Close the Serial Monitor.

## 2. Run the viewer
    python3 -m pip install pyserial opencv-python numpy
    python3 viewer.py /dev/cu.usbmodem1101

Press q in the video window to quit.
If the image is upside down, change set_vflip(s, 1) to set_vflip(s, 0) in the sketch.
