#include <Wire.h>
#include <MPU6050.h>
#include <ESP32Servo.h>

MPU6050 mpu;
Servo myServo;

// ---------------- PIN ASSIGNMENTS ----------------
const int VIBRATION_PIN = 4;    // Piezo sensor (analog)
const int SERVO_PIN     = 18;   // Servo signal (PWM)
// MPU6050 -> SDA = GPIO21, SCL = GPIO22 (default I2C, no explicit pin needed)

unsigned long lastSend = 0;
const int SEND_INTERVAL = 100;   // ~10 readings/sec

// --- Gentle mode settings (default/baseline servo behavior) ---
int currentAngle = 90;
int targetAngle = 90;
unsigned long lastMove = 0;
const int MOVE_INTERVAL = 30;
const int STEP_SIZE = 1;
unsigned long lastTargetChange = 0;
const int TARGET_CHANGE_INTERVAL = 2000;

// --- Jerk burst mode settings (triggered live during demo) ---
bool jerkMode = false;
unsigned long jerkStartTime = 0;
const int JERK_BURST_DURATION = 1500;
unsigned long lastJerk = 0;
const int JERK_INTERVAL = 100;

void setup() {
  Serial.begin(115200);
  Wire.begin();               // SDA=21, SCL=22 by default
  mpu.initialize();

  myServo.setPeriodHertz(50);
  myServo.attach(SERVO_PIN, 500, 2400);
  myServo.write(currentAngle);
  randomSeed(analogRead(0));

  if (!mpu.testConnection()) {
    Serial.println("ERROR: MPU6050 not found");
  } else {
    Serial.println("MPU6050 connected. System ready.");
  }
}

void loop() {
  // --- Check for jerk-trigger command from Python ('J') ---
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'J') {
      jerkMode = true;
      jerkStartTime = millis();
    }
  }

  // --- Servo behavior: jerk mode overrides gentle mode ---
  if (jerkMode) {
    if (millis() - jerkStartTime < JERK_BURST_DURATION) {
      if (millis() - lastJerk >= JERK_INTERVAL) {
        lastJerk = millis();
        int randAngle = random(30, 151);   // wider, sharper jumps
        myServo.write(randAngle);
      }
    } else {
      jerkMode = false;
      currentAngle = 90;
      myServo.write(currentAngle);
    }
  } else {
    if (millis() - lastTargetChange >= TARGET_CHANGE_INTERVAL) {
      lastTargetChange = millis();
      targetAngle = 90 + random(-15, 16);  // gentle narrow range
    }
    if (millis() - lastMove >= MOVE_INTERVAL) {
      lastMove = millis();
      if (currentAngle < targetAngle) currentAngle += STEP_SIZE;
      else if (currentAngle > targetAngle) currentAngle -= STEP_SIZE;
      myServo.write(currentAngle);
    }
  }

  // --- Sensor reading + send ---
  if (millis() - lastSend >= SEND_INTERVAL) {
    lastSend = millis();

    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    float tilt = atan2(ay, az) * 180.0 / PI;

    int vibration = analogRead(VIBRATION_PIN);

    // CSV format: tilt,vibration,timestamp
    Serial.print(tilt, 2);
    Serial.print(",");
    Serial.print(vibration);
    Serial.print(",");
    Serial.println(millis());
  }
}