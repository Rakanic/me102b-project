// ESP32S3-CAM: stream JPEG frames over USB serial (no Wi-Fi)
// Board settings: same as before (ESP32S3 Dev Module, USB CDC On Boot: Enabled,
// PSRAM: OPI PSRAM, USB Mode: Hardware CDC and JTAG).
// Tip: Tools -> Core Debug Level -> None, so log text doesn't mix into the stream.

#include "esp_camera.h"

// Pin map for this board (identical to CAMERA_MODEL_ESP32S3_EYE)
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  15
#define SIOD_GPIO_NUM   4
#define SIOC_GPIO_NUM   5
#define Y2_GPIO_NUM    11
#define Y3_GPIO_NUM     9
#define Y4_GPIO_NUM     8
#define Y5_GPIO_NUM    10
#define Y6_GPIO_NUM    12
#define Y7_GPIO_NUM    18
#define Y8_GPIO_NUM    17
#define Y9_GPIO_NUM    16
#define VSYNC_GPIO_NUM  6
#define HREF_GPIO_NUM   7
#define PCLK_GPIO_NUM  13

// ---- Fixed camera settings (needed for colour recognition) ----
// Auto exposure and auto white balance shift colours from frame to frame and
// scene to scene, which would invalidate the colour calibration. With
// LOCK_CAMERA 1 the camera uses fixed values instead. Tune these once for
// your lighting until the image looks right, calibrate, and then leave them
// alone. If you change them (or the lighting), recalibrate.
#define LOCK_CAMERA 1
#define WB_MODE     1    // white balance preset: 1 sunny, 2 cloudy, 3 office, 4 home
#define AEC_VALUE   300  // exposure, 0-1200: raise if the image is too dark
#define AGC_GAIN    0    // analog gain, 0-30: raise only if still too dark at max exposure

// Each frame is sent as: "FRAM" + 4-byte little-endian length + JPEG bytes
static const uint8_t MAGIC[4] = {'F', 'R', 'A', 'M'};

void setup() {
  Serial.begin(115200);          // baud is ignored for native USB
  Serial.setDebugOutput(false);  // keep ESP log messages out of the stream

  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk  = XCLK_GPIO_NUM;
  config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href  = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn  = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_VGA;   // 320x240; small frames = higher fps over USB
  config.jpeg_quality = 8;               // 0-63, lower = better quality, bigger frames
  config.fb_count     = 2;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    while (true) {
      Serial.printf("Camera init failed: 0x%x\n", err);
      delay(1000);
    }
  }

  sensor_t *s = esp_camera_sensor_get();
  s->set_vflip(s, 1);  // set to 0 if the image comes out upside down

#if LOCK_CAMERA
  s->set_wb_mode(s, WB_MODE);     // fixed white-balance preset instead of auto
  s->set_exposure_ctrl(s, 0);     // auto exposure off
  s->set_aec_value(s, AEC_VALUE);
  s->set_gain_ctrl(s, 0);         // auto gain off
  s->set_agc_gain(s, AGC_GAIN);
#endif
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;

  uint32_t len = fb->len;
  Serial.write(MAGIC, 4);
  Serial.write((const uint8_t *)&len, 4);
  Serial.write(fb->buf, fb->len);

  esp_camera_fb_return(fb);
}
