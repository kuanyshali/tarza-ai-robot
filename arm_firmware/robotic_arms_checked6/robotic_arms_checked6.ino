/*
  ======================================================================
   TWO ROBOTIC ARMS  -  POSITION CONTROL  -  Arduino Mega
  ======================================================================
   Each arm:  stepper joint (DM542)  + potentiometer   (position feedback)
              elbow servo (MG996R)   + potentiometer   (position feedback)
              gripper servo (DS329MG)                  (open / close)
   You move the joints to SAVED POSITIONS (0 = zero, 1..4) instead of degrees.
   The positions are pot readings, the same numbers "p" shows.

   Library needed: AccelStepper   (Servo and EEPROM are built in)

  ------------------------------- WIRING -------------------------------
                            RIGHT arm      LEFT arm
   Stepper PUL-             pin 6          pin 5
   Stepper DIR-             pin 7          pin 4
   Stepper PUL+ / DIR+      Arduino 5V     Arduino 5V
   Stepper pot (middle)     A0             A4
   Elbow servo signal       pin 9          pin 13
   Elbow pot (middle)       none           A6
   Hand / Gripper signal    pin 10 (hand)  pin 11 (gripper)
   (Left elbow servo is on pin 12.)

   Pots:     outer pins -> Arduino 5V and Arduino GND.
   Steppers: driver +V/GND -> 24-30V supply. Driver GND NOT to Arduino (common anode).
   Servos:   middle wire (red/orange) -> 5-6V +, brown -> supply GND.
             Each servo supply GND -> Arduino GND with its own wire (star ground).

   !! Switch all motor supplies ON BEFORE the Arduino starts.
   START-UP (safe, no fast moves):
      - Elbow calibration is SAVED (EEPROM): at power-on elbows start where they are.
      - Only the FIRST time (or after "s c") an elbow calibrates, SLOWLY 85 -> 100 deg.
      - Steppers hold where they are. Grippers don't move until "g o" / "g c".

  ------------------------------ COMMANDS ------------------------------
   Serial Monitor: 115200 baud, line ending "New Line". Spaces are optional.
   Pick the arm first:  r = RIGHT   l = LEFT   b = BOTH   (no letter = RIGHT)

   WHOLE ARM   2        stepper AND elbow to position 2      (0 = zero position)
               l 0      left arm to zero position
               b 1      both arms to position 1

   STEPPER     m 3      stepper to position 3
   ELBOW       s 1      elbow to position 1
   RIGHT ELBOW has NO pot: positions by timing (same slow speed, n x R_EL_STEP_MS up).
               At power-on it must hang DOWN (= position 0) and it holds there.
               "0" / "r 0" / "b 0" = arm to zero (hand stops shaking). Elbow keeps holding.
               "s a 60" = elbow to servo angle 60 (use it to find R_EL_REST).

   HAND        g s      right hand SHAKE (both sides, again and again)
               g n      right hand NO SHAKE (goes to the center, then its servo turns OFF)
               g f      hand / gripper servo OFF right now (no signal, no shaking)
   GRIPPER     l g o    left gripper open     l g c   left gripper close
               l g f    left gripper servo OFF right now     b g f  both OFF
               g a 150  hand / gripper to any angle (for tuning)

   FINE MOVES  m l r 5    stepper +5 deg   (relative)     m a 90   stepper to pot reading 90
               s r -5   elbow -5 deg     (relative)     s a 120  elbow to pot reading 120

   SPEED       m v 20   stepper speed (deg/s)   m c 40  stepper accel   s v 30  elbow speed
   JOG         l m f    left stepper rotates FORWARD  until you stop it (m f = right)
               l m b    left stepper rotates BACKWARD until you stop it (m b = right)
   STOP        m x / s x   stop one joint       x   EMERGENCY STOP everything
   OTHER       p   status     t   positions table     s c   re-calibrate elbow     h   help

   Examples:  b 0      both arms to zero        l m 2    left stepper to position 2
              r s 4    right elbow to pos 4     g s      right hand shake
  ======================================================================
*/

#include <AccelStepper.h>
#include <Servo.h>
#include <EEPROM.h>

// ============================ SETTINGS (EDIT) ============================
// ---- Pins ----
const int R_PUL_PIN = 6,  R_DIR_PIN = 7,  R_M_POT = A0;
const int R_SERVO_PIN = 9,  R_S_POT = -1,  R_GRIP_PIN = 10;   // right elbow: NO pot

const int L_PUL_PIN = 5,  L_DIR_PIN = 4,  L_M_POT = A4;
const int L_SERVO_PIN = 13, L_S_POT = A6,  L_GRIP_PIN = 12;

// ---- Stepper direction ----
// If a stepper gives "moving the wrong way", flip that arm's M_DIR_INVERT.
const bool R_M_POT_INVERT = false,  R_M_DIR_INVERT = false;
const bool L_M_POT_INVERT = false,  L_M_DIR_INVERT = false;

// ---- Elbow pot direction ----
// false = elbow readings are the same numbers as the pot reader (= your positions table).
// If the elbows go the WRONG WAY through the positions, flip this.
// Changing it makes the elbows re-calibrate once (slowly) at the next start.
const bool S_POT_INVERT = false;

// ---- POSITIONS (pot readings, from your table) ----
// Index:                0 = zero    1      2      3      4       NONE = not set
const float NONE = -1;
const float R_M_POS[5] = {  250,      220,   190,   160,   140 };   // right stepper
// ---- RIGHT ELBOW: NO POT, POSITIONS BY TIMING ----
// The elbow always moves at the same slow speed (R_EL_SPEED).
// Position n = moving UP from rest for n x R_EL_STEP_MS milliseconds.
// At power-on the elbow must hang DOWN (rest = position 0).
const float R_EL_REST    = 60;     // EDIT: servo angle when the arm hangs DOWN (find it with "s a")
const int   R_EL_UP      = +1;     // EDIT: +1 = bigger angle is UP, -1 = smaller angle is UP
const float R_EL_SPEED   = 20.0;   // deg/s the elbow moves (slow and smooth)
const unsigned long R_EL_STEP_MS = 1300;   // time per position (1300 ms x 20 deg/s = 26 deg)
const float R_EL_STEP    = R_EL_SPEED * R_EL_STEP_MS / 1000.0;
const float R_S_ANG[5] = { R_EL_REST,
                           R_EL_REST + R_EL_UP * R_EL_STEP * 1,
                           R_EL_REST + R_EL_UP * R_EL_STEP * 2,
                           R_EL_REST + R_EL_UP * R_EL_STEP * 3,
                           R_EL_REST + R_EL_UP * R_EL_STEP * 4 };   // = 30, 56, 82, 108, 134
const float L_M_POS[5] = {  52,      82,   112,   142,   172 };   // left stepper
const float L_S_POS[5] = { 144,     174,   204,   234,   264 };   // left elbow

// ---- LIMITS (pot readings, from your table) ----
// A joint is never sent outside these (MIN = smaller number, MAX = bigger number).
const float R_M_MIN =  280, R_M_MAX = 130;    // right stepper (your min 10 / max 178)
const float L_M_MIN =  13, L_M_MAX = 184.5;  // left stepper  (your min 13 / max 184.5)
const float L_S_MIN =  91, L_S_MAX = 280;    // left elbow    (your min 91 / max 280)

// ---- Gripper angles ----
// Bigger CLOSE = tighter. Find exact values live with "g a <angle>" / "l g a <angle>".
const int R_GRIP_OPEN = 4, R_GRIP_CLOSE = 100;
const int L_GRIP_OPEN = 140, L_GRIP_CLOSE = 80;
const float GRIP_SPEED = 60.0;        // gripper deg/s (lower = smoother and slower, e.g. 40)
// true  = after each move the gripper RELAXES (signal off): quiet, no twitching.
//         The metal gears still hold it, but a closed grip is weaker.
// false = the gripper keeps pushing all the time (strong grip, may twitch/buzz).
const bool GRIP_RELAX = true;

// ---- RIGHT HAND (installed instead of the right gripper, same servo, pin 10) ----
// "g s" = SHAKE: rotates to one side, then the other, again and again
// "g n" = NO SHAKE: stops and goes back to the center
const bool  R_HAS_HAND  = true;       // false = right arm has a gripper again (g o / g c)
const int   HAND_CENTER = 90;         // resting angle of the hand
const int   HAND_SWING  = 30;         // how far to each side -> shakes between 60 and 120
const float HAND_SPEED  = 90.0;       // deg/s while shaking (lower = slower shake)
const bool  HAND_RELAX  = false;      // false = hand keeps holding after it stops (doesn't fall)

// ---- Default speeds ----
const float M_SPEED_DEFAULT = 15.0;   // stepper deg/s
const float M_ACCEL_DEFAULT = 30.0;   // stepper deg/s^2
const float S_SPEED_DEFAULT = 30.0;   // elbow deg/s

// ---- Jog ("l m f" forward / "l m b" backward, until "l m x" or "x") ----
// Forward  = the way the positions go (from zero toward position 4 and past it).
// Backward = the other way (from position 4 toward zero and past it).
// Speed = the stepper speed ("l m v 10" to change it).
const bool JOG_STOP_AT_LIMIT = true;  // true = stops by itself at the joint's limit

// ---- Mechanics ----
const float POT_RANGE_DEG = 300.0;    // pot full rotation
const float STEPS_PER_REV = 1600.0;   // must match DM542 SW5-SW8
const float GEAR_RATIO    = 1.0;
const float STEPS_PER_DEG = STEPS_PER_REV * GEAR_RATIO / 360.0;

// ---- Tuning (normally leave as is) ----
const float M_DEADBAND  = 0.8;
const float M_RESTART   = 2.0;
const float M_KP        = 2.0;
const float M_MIN_SPEED = 2.0;
const float CAL_A = 85.0, CAL_B = 100.0;   // elbow calibration servo angles (small move)
const float CAL_SPEED = 10.0;              // deg/s during calibration
const bool  USE_SAVED_CAL = true;          // start elbows from saved calibration (no jump)
const float S_DEADBAND  = 1.5;   // in real pot degrees (scaled automatically for the right elbow)
const float S_RESTART   = 3.0;   // in real pot degrees (scaled automatically for the right elbow)
const int   S_MAX_TRIES = 6;
const unsigned long S_SETTLE_MS = 300;
// ========================================================================

struct Arm {
  const char *name;
  // stepper
  AccelStepper *stepper;
  int   mPotPin;
  bool  mPotInvert, mDirInvert;
  float mMin, mMax;
  const float *mPos;
  float mAngle, mTarget, mSpeedNow, mStartErr;
  bool  mMoving, mFault;
  bool  mJog;                          // true while jogging
  float mJogStart, mJogDir;            // mJogDir: +1 / -1 in pot degrees
  unsigned long mLast;
  float mSpeedDeg, mAccelDeg;
  // elbow
  Servo servo;
  int   sPin, sPotPin;
  float sMin, sMax;
  const float *sPos;
  bool  sCalOK;
  float sK, sB, sCmd, sCmdGoal, sAngle, sTarget;
  bool  sMoving, sQuiet;
  int   sTries;
  unsigned long sLast, sSettleStart, sLastHold;
  float sSpeedDeg;
  // gripper
  Servo gripper;
  int   gPin, gOpenDeg, gCloseDeg;
  bool  gripOpen;
  bool  isHand, shaking;               // isHand: a shaking hand instead of a gripper
  bool  sNoPot;                        // elbow without pot: works by servo angles only
  unsigned long sArrived;              // when the no-pot elbow reached its goal
  float gCmd, gGoal;                   // gripper angle now / goal (moves smoothly)
  bool  gStarted;                      // true after the first g o / g c
  bool  gOffWhenDone;                  // true = turn the servo OFF when it arrives (g n)
  unsigned long gDone;                 // when the last move finished
  unsigned long gLast;
};

AccelStepper stepR(AccelStepper::DRIVER, R_PUL_PIN, R_DIR_PIN);
AccelStepper stepL(AccelStepper::DRIVER, L_PUL_PIN, L_DIR_PIN);
Arm armR, armL;

// ============================ HELPERS ============================
void initArm(Arm &a, const char *name, AccelStepper *st,
             int mPot, bool potInv, bool dirInv, float mMin, float mMax, const float *mPos,
             int sPin, int sPot, float sMin, float sMax, const float *sPos,
             int gPin, int gOpen, int gClose) {
  a.name = name;
  a.stepper = st; a.mPotPin = mPot; a.mPotInvert = potInv; a.mDirInvert = dirInv;
  // Safety: if MIN and MAX were typed the wrong way round, swap them
  if (mMin > mMax) { float t = mMin; mMin = mMax; mMax = t; }
  if (sMin > sMax) { float t = sMin; sMin = sMax; sMax = t; }
  a.mMin = mMin; a.mMax = mMax; a.mPos = mPos;
  a.mAngle = a.mTarget = a.mSpeedNow = a.mStartErr = 0;
  a.mMoving = a.mFault = false; a.mLast = 0;
  a.mJog = false; a.mJogStart = 0; a.mJogDir = 0;
  a.mSpeedDeg = M_SPEED_DEFAULT; a.mAccelDeg = M_ACCEL_DEFAULT;

  a.sPin = sPin; a.sPotPin = sPot; a.sMin = sMin; a.sMax = sMax; a.sPos = sPos;
  a.sCalOK = false; a.sK = 1.0; a.sB = 0; a.sCmd = a.sCmdGoal = 90;
  a.sAngle = a.sTarget = 0;
  a.sMoving = a.sQuiet = false; a.sTries = 0;
  a.sLast = a.sSettleStart = a.sLastHold = 0;
  a.sSpeedDeg = S_SPEED_DEFAULT;

  a.gPin = gPin; a.gOpenDeg = gOpen; a.gCloseDeg = gClose; a.gripOpen = true;
  a.gCmd = a.gGoal = gOpen; a.gLast = 0; a.gStarted = false; a.gDone = 0;
  a.gOffWhenDone = false;
  a.isHand = false; a.shaking = false;
  a.sNoPot = false; a.sArrived = 0;
}

void tag(Arm &a) { Serial.print(a.name); Serial.print(F(" ")); }


float readPotDeg(int pin, bool invert) {
  long sum = 0;
  for (int i = 0; i < 4; i++) sum += analogRead(pin);
  float deg = (sum / 4.0) * POT_RANGE_DEG / 1023.0;
  if (invert) deg = POT_RANGE_DEG - deg;
  return deg;
}

float readPotSlow(int pin, bool invert) {
  float s = 0;
  for (int i = 0; i < 20; i++) { s += readPotDeg(pin, invert); delay(5); }
  return s / 20.0;
}

void writeServo(Arm &a) {
  a.servo.writeMicroseconds(500 + (int)(a.sCmd * 2000.0 / 180.0));
}

// ============================ STEPPER ============================
void stepperGoTo(Arm &a, float deg) {
  if (deg < a.mMin || deg > a.mMax) {
    deg = constrain(deg, a.mMin, a.mMax);
    tag(a); Serial.print(F("stepper: limit is ")); Serial.print(a.mMin);
    Serial.print(F(" to ")); Serial.print(a.mMax); Serial.println(F(", going to the nearest limit."));
  }
  a.mJog = false;                      // a new target ends jogging
  a.mTarget = deg;
  a.mMoving = true;
  a.mFault = false;
  a.mStartErr = fabs(a.mTarget - a.mAngle);
  tag(a); Serial.print(F("stepper -> ")); Serial.println(deg);
}

void stepperHold(Arm &a) {
  a.mTarget = a.mAngle; a.mMoving = false; a.mFault = false; a.mJog = false;
}

// Direction of "backward" in pot degrees: from position 4 toward zero (+1 or -1)
float backDir(Arm &a) {
  int last = 4;
  while (last > 0 && a.mPos[last] == NONE) last--;
  return (last == 0 || a.mPos[0] >= a.mPos[last]) ? 1 : -1;
}

// Starts continuous rotation: forward = toward the positions, backward = the other way
void stepperJog(Arm &a, bool forward) {
  a.mJog = true; a.mMoving = false; a.mFault = false;
  a.mJogStart = a.mAngle;
  a.mJogDir = forward ? -backDir(a) : backDir(a);
  tag(a); Serial.print(F("stepper rotating "));
  Serial.print(forward ? F("FORWARD") : F("BACKWARD"));
  Serial.println(F("... send 'm x' (or 'x') to stop."));
}

void stepperFault(Arm &a, const __FlashStringHelper *msg) {
  a.mFault = true; a.mMoving = false; a.mSpeedNow = 0; a.mJog = false;
  a.stepper->setSpeed(0);
  tag(a); Serial.print(F("STEPPER STOPPED: ")); Serial.println(msg);
}

void updateStepper(Arm &a) {
  unsigned long now = millis();
  if (now - a.mLast < 10) return;
  float dt = (now - a.mLast) / 1000.0;
  a.mLast = now;

  float raw = readPotDeg(a.mPotPin, a.mPotInvert);
  a.mAngle = 0.6 * a.mAngle + 0.4 * raw;            // smooth pot noise
  float err = a.mTarget - a.mAngle;
  float desired = 0;

  if (a.mJog) {                                      // jogging (forward or backward)
    float dir = a.mJogDir;
    if (JOG_STOP_AT_LIMIT && ((dir > 0 && a.mAngle >= a.mMax) || (dir < 0 && a.mAngle <= a.mMin))) {
      stepperHold(a);
      tag(a); Serial.print(F("jog stopped: limit reached at ")); Serial.println(a.mAngle);
    } else if (dir * (a.mAngle - a.mJogStart) < -10.0) {
      stepperFault(a, F("jog is moving the wrong way. Flip this arm's M_DIR_INVERT."));
      return;
    } else {
      desired = dir * a.mSpeedDeg;
    }
  } else if (!a.mFault) {
    if (!a.mMoving && fabs(err) > M_RESTART) {       // drifted (gravity) -> correct
      a.mMoving = true;
      a.mStartErr = fabs(err);
    }
    if (a.mMoving) {
      if (fabs(err) <= M_DEADBAND) {
        a.mMoving = false;
        tag(a); Serial.print(F("stepper reached ")); Serial.println(a.mAngle);
      } else if (fabs(err) > a.mStartErr + 15.0) {
        stepperFault(a, F("moving the wrong way. Flip this arm's M_DIR_INVERT."));
        return;
      } else if ((a.mAngle < a.mMin - 5 && err < 0) || (a.mAngle > a.mMax + 5 && err > 0)) {
        stepperFault(a, F("joint outside limits."));
        return;
      } else {
        desired = M_KP * err;
        if (fabs(desired) < M_MIN_SPEED) desired = (err > 0) ? M_MIN_SPEED : -M_MIN_SPEED;
        desired = constrain(desired, -a.mSpeedDeg, a.mSpeedDeg);
      }
    }
  }

  float maxChange = a.mAccelDeg * dt;                 // smooth acceleration
  a.mSpeedNow += constrain(desired - a.mSpeedNow, -maxChange, maxChange);
  if (!a.mMoving && !a.mJog && fabs(a.mSpeedNow) < 0.5) a.mSpeedNow = 0;

  a.stepper->setSpeed((a.mDirInvert ? -1 : 1) * a.mSpeedNow * STEPS_PER_DEG);
}

// ============================ ELBOW ============================
// ---- Saved calibration (EEPROM): right arm at address 0, left at 16 ----
struct CalData { uint16_t magic; float k, b; };
// The magic number depends on S_POT_INVERT, so a calibration saved with the
// other setting is ignored and the elbow re-calibrates (prevents wrong moves).
// Changed after the new sensor readings (Nov 12): old saved calibrations are
// ignored, so both elbows re-calibrate once (slowly) at the next start.
const uint16_t CAL_MAGIC = S_POT_INVERT ? 0xA56A : 0xA56B;

// Left arm moved to address 32 after its parts were changed: its old saved
// calibration is ignored, so the left elbow re-calibrates once (slowly).
// Save slots are moved after new elbow readings (right: Nov 13 -> 48, left: Nov 13 -> 64),
// so an old saved calibration is ignored and that elbow re-calibrates once (slowly).
// Right elbow moved to address 80 after the x2.65 scaling (Nov 23): its old saved
// calibration (without scaling) is ignored, so it re-calibrates once (slowly).
int calAddr(Arm &a) { return (&a == &armR) ? 80 : 64; }

void saveCal(Arm &a) {
  CalData d = { CAL_MAGIC, a.sK, a.sB };
  EEPROM.put(calAddr(a), d);
}

bool loadCal(Arm &a) {
  CalData d;
  EEPROM.get(calAddr(a), d);
  float sc = 1.0;
  if (d.magic != CAL_MAGIC || isnan(d.k) || isnan(d.b) || fabs(d.k) < 0.3 * sc || fabs(d.k) > 3.0 * sc) return false;
  a.sK = d.k; a.sB = d.b;
  return true;
}

// Moves the elbow servo SLOWLY to an angle (used only in calibration)
void rampServo(Arm &a, float target, float degPerSec) {
  float step = degPerSec * 0.02;            // per 20 ms
  while (fabs(target - a.sCmd) > step) {
    a.sCmd += (target > a.sCmd) ? step : -step;
    writeServo(a);
    delay(20);
  }
  a.sCmd = target;
  writeServo(a);
}

void calibrateElbow(Arm &a) {
  tag(a); Serial.println(F("elbow calibration: moving slowly 85 -> 100 deg..."));
  rampServo(a, CAL_A, CAL_SPEED); delay(600);
  float pa = readPotSlow(a.sPotPin, S_POT_INVERT);
  rampServo(a, CAL_B, CAL_SPEED); delay(600);
  float pb = readPotSlow(a.sPotPin, S_POT_INVERT);

  float k = (pb - pa) / (CAL_B - CAL_A);
  Serial.print(F("  pot at 85: ")); Serial.print(pa);
  Serial.print(F("   pot at 100: ")); Serial.print(pb);
  Serial.print(F("   ratio: ")); Serial.println(k, 3);

  a.sCmdGoal = a.sCmd;
  a.sAngle = a.sTarget = pb;
  a.sMoving = false;

  if (fabs(k) < 0.3) {
    a.sCalOK = false;
    a.sK = 1.0; a.sB = 0.0;
    Serial.println(F("  !! The pot did NOT follow the servo (changed < 4.5 deg for a 15 deg move)."));
    Serial.println(F("     Check: servo power ON + current up, shared GND, signal pin,"));
    Serial.println(F("     pot middle pin, pot shaft turns with the joint, servo strong enough."));
    Serial.println(F("     Elbow positions are DISABLED until this works. Fix, then 's c'."));
    return;
  }
  if (fabs(k) > 3.0) Serial.println(F("  !! Ratio looks too big - pot reading is noisy. Check pot wiring / GND."));
  a.sK = k;
  a.sB = pa - k * CAL_A;
  a.sCalOK = true;
  saveCal(a);
  Serial.println(F("  OK: elbow feedback active. Calibration saved (next start: no movement)."));
}

// Start-up of one elbow: no jump if a saved calibration exists
void startElbow(Arm &a) {
  if (a.sNoPot) {
    // No pot: the elbow is ASSUMED to hang DOWN at power-on. The servo starts right
    // there (rest angle) and holds it, so if R_EL_REST is right, nothing moves.
    a.sCmd = a.sCmdGoal = a.sTarget = a.sAngle = R_EL_REST;
    a.sMoving = false;
    a.sCalOK = true;
    a.sSpeedDeg = R_EL_SPEED;
    writeServo(a);                            // pulse set BEFORE attach -> starts at rest
    a.servo.attach(a.sPin, 500, 2500);
    tag(a); Serial.print(F("elbow (no pot): starting at REST (down) = "));
    Serial.print(R_EL_REST); Serial.println(F(" deg, holding."));
    return;
  }

  float pot = readPotSlow(a.sPotPin, S_POT_INVERT);

  if (USE_SAVED_CAL && loadCal(a)) {
    float cmd = (pot - a.sB) / a.sK;          // servo angle the elbow is at right now
    if (cmd >= 0 && cmd <= 180) {
      a.sCmd = a.sCmdGoal = cmd;
      writeServo(a);                          // set the pulse BEFORE attach -> no jump
      a.servo.attach(a.sPin, 500, 2500);
      a.sCalOK = true;
      a.sAngle = a.sTarget = pot;
      a.sMoving = false;
      tag(a); Serial.print(F("elbow: saved calibration, starting where it is (pot "));
      Serial.print(pot); Serial.println(F("). Send 's c' to recalibrate."));
      return;
    }
    tag(a); Serial.println(F("elbow: saved calibration doesn't match the current position, recalibrating."));
  }

  a.sCmd = a.sCmdGoal = 90;                   // first time: one position, then slow calibration
  writeServo(a);
  a.servo.attach(a.sPin, 500, 2500);
  delay(800);
  calibrateElbow(a);
}

void servoGoTo(Arm &a, float deg) {
  if (a.sNoPot) {                                     // elbow without pot: deg = servo angle
    if (deg < a.sMin || deg > a.sMax) {
      deg = constrain(deg, a.sMin, a.sMax);
      tag(a); Serial.println(F("elbow: angle limited to 0..180."));
    }
    if (!a.servo.attached()) {                        // switched off -> on, where it is (no jump)
      writeServo(a);
      a.servo.attach(a.sPin, 500, 2500);
    }
    a.sCmdGoal = a.sTarget = deg;
    a.sMoving = true;
    tag(a); Serial.print(F("elbow -> servo ")); Serial.print(deg); Serial.println(F(" deg"));
    return;
  }
  if (!a.sCalOK) {
    tag(a); Serial.println(F("elbow: feedback is OFF, can't use positions. Fix the elbow, then 's c'."));
    return;
  }
  if (deg < a.sMin || deg > a.sMax) {
    deg = constrain(deg, a.sMin, a.sMax);
    tag(a); Serial.print(F("elbow: limit is ")); Serial.print(a.sMin);
    Serial.print(F(" to ")); Serial.print(a.sMax); Serial.println(F(", going to the nearest limit."));
  }
  float goal = (deg - a.sB) / a.sK;                   // servo angle for this pot reading
  if (goal < 0 || goal > 180) {
    goal = constrain(goal, 0, 180);
    deg = a.sB + a.sK * goal;
    tag(a); Serial.print(F("elbow: servo can only reach ")); Serial.print(deg);
    Serial.println(F(" in that direction, going there."));
  }
  a.sTarget = deg;
  a.sCmdGoal = goal;
  a.sMoving = true;
  a.sQuiet = false;
  a.sTries = 0;
  tag(a); Serial.print(F("elbow -> ")); Serial.println(deg);
}

void servoHold(Arm &a) {
  if (a.sNoPot) { a.sCmdGoal = a.sTarget = a.sAngle = a.sCmd; a.sMoving = false; return; }
  a.sTarget = a.sAngle; a.sCmdGoal = a.sCmd; a.sMoving = false;
}

// Elbow without pot: moves the servo smoothly (sSpeedDeg) to its goal angle.
// It keeps holding at every position, also at zero.
void updateServoNoPot(Arm &a, unsigned long now, float dt) {
  if (!a.servo.attached()) return;                     // servo is off
  float diff = a.sCmdGoal - a.sCmd;
  if (fabs(diff) > 0.01) {
    float step = a.sSpeedDeg * dt;
    a.sCmd = (fabs(diff) <= step) ? a.sCmdGoal : a.sCmd + (diff > 0 ? step : -step);
    a.sAngle = a.sCmd;
    writeServo(a);
    a.sArrived = now;
    return;
  }
  if (a.sMoving) {
    a.sMoving = false;
    tag(a); Serial.print(F("elbow reached servo ")); Serial.print(a.sCmd); Serial.println(F(" deg"));
  }
}

void updateServo(Arm &a) {
  unsigned long now = millis();
  if (now - a.sLast < 20) return;
  float dt = (now - a.sLast) / 1000.0;
  a.sLast = now;
  if (a.sNoPot) { updateServoNoPot(a, now, dt); return; }

  float raw = readPotDeg(a.sPotPin, S_POT_INVERT);
  a.sAngle = 0.6 * a.sAngle + 0.4 * raw;

  float diff = a.sCmdGoal - a.sCmd;                    // move smoothly toward goal
  if (fabs(diff) > 0.01) {
    float maxStep = a.sSpeedDeg * dt / fabs(a.sK);
    if (fabs(diff) <= maxStep) a.sCmd = a.sCmdGoal;
    else a.sCmd += (diff > 0 ? maxStep : -maxStep);
    writeServo(a);
    a.sSettleStart = now;
    return;
  }

  if (!a.sCalOK) return;
  if (now - a.sSettleStart < S_SETTLE_MS) return;     // let the servo arrive

  float err = a.sTarget - a.sAngle;
  if (a.sMoving) {
    if (fabs(err) <= S_DEADBAND) {
      a.sMoving = false;
      if (!a.sQuiet) { tag(a); Serial.print(F("elbow reached ")); Serial.println(a.sAngle); }
      return;
    }
    float newGoal = constrain(a.sCmdGoal + err / a.sK, 0, 180);
    if (a.sTries >= S_MAX_TRIES || fabs(newGoal - a.sCmdGoal) < 0.2) {
      a.sMoving = false;
      if (!a.sQuiet) {
        tag(a); Serial.print(F("elbow stopped at ")); Serial.print(a.sAngle);
        Serial.print(F(" (target ")); Serial.print(a.sTarget);
        Serial.println(F("). Servo can't get closer: too much load or at its end."));
      }
      return;
    }
    a.sCmdGoal = newGoal;                              // small correction
    a.sTries++;
  } else if (fabs(err) > S_RESTART && now - a.sLastHold > 1000) {
    a.sLastHold = now;                                 // drifted -> quietly correct
    a.sMoving = true;
    a.sQuiet = true;
    a.sTries = 0;
  }
}

// ============================ GRIPPER ============================
// Same pulse range as Servo.write(): 0 deg = 544 us, 180 deg = 2400 us,
// but with fractions of a degree, so small steps are smooth
void writeGripper(Arm &a) {
  a.gripper.writeMicroseconds(544 + (int)(a.gCmd * (2400 - 544) / 180.0));
}

// Turns the gripper signal on at the angle it is at now (no jump)
void gripperAttach(Arm &a) {
  if (a.gripper.attached()) return;
  writeGripper(a);                                     // set angle BEFORE attach
  a.gripper.attach(a.gPin);
}

// The gripper servo is started at the first g o / g c (no movement at power-on).
// It has no pot, so the first time it goes straight to that angle; after that it moves smoothly.
void gripperStart(Arm &a, int deg) {
  if (!a.gStarted) {
    a.gStarted = true;
    a.gCmd = a.gGoal = deg;
  }
  gripperAttach(a);                                    // re-powers a relaxed gripper
  a.gOffWhenDone = false;                              // a new command keeps it ON until told
  a.gLast = a.gDone = millis();
}

void gripperOpen(Arm &a) {
  gripperStart(a, a.gOpenDeg);
  a.gGoal = a.gOpenDeg; a.gripOpen = true;
  tag(a); Serial.println(F("gripper OPEN"));
}
void gripperClose(Arm &a) {
  gripperStart(a, a.gCloseDeg);
  a.gGoal = a.gCloseDeg; a.gripOpen = false;
  tag(a); Serial.println(F("gripper CLOSE"));
}

// ---- Hand (right arm): shake / no shake ----
void handShake(Arm &a) {
  gripperStart(a, HAND_CENTER);
  a.shaking = true;
  a.gGoal = HAND_CENTER + HAND_SWING;                  // first swing, then it alternates
  tag(a); Serial.println(F("hand SHAKING... send 'g n' to stop"));
}

void handStill(Arm &a) {
  gripperStart(a, HAND_CENTER);
  a.shaking = false;
  a.gGoal = HAND_CENTER;                               // back to the middle
  a.gOffWhenDone = true;                               // then the servo turns OFF
  tag(a); Serial.println(F("hand NOT shaking, back to center, then servo OFF"));
}

// "g f": hand / gripper servo OFF right now (signal off, it stops where it is)
void gripperOff(Arm &a) {
  a.shaking = false;
  a.gGoal = a.gCmd;                                    // no move left to do
  a.gOffWhenDone = false;
  if (a.gripper.attached()) a.gripper.detach();
  tag(a); Serial.println(a.isHand ? F("hand servo OFF") : F("gripper servo OFF"));
}

// Moves the gripper / hand a small step toward its goal every 20 ms.
// Hand shaking: at each side the goal flips to the other side, so it keeps rotating both ways.
// Gripper with GRIP_RELAX: signal off 0.4 s after it stops (no twitching). The hand keeps holding.
void updateGripper(Arm &a) {
  if (!a.gStarted || !a.gripper.attached()) return;
  unsigned long now = millis();
  if (now - a.gLast < 20) return;
  float dt = (now - a.gLast) / 1000.0;
  a.gLast = now;
  float diff = a.gGoal - a.gCmd;
  if (fabs(diff) >= 0.01) {
    float step = (a.shaking ? HAND_SPEED : GRIP_SPEED) * dt;
    a.gCmd = (fabs(diff) <= step) ? a.gGoal : a.gCmd + (diff > 0 ? step : -step);
    writeGripper(a);
    a.gDone = now;
  } else if (a.shaking) {
    a.gGoal = (a.gGoal > HAND_CENTER) ? HAND_CENTER - HAND_SWING   // reached one side:
                                      : HAND_CENTER + HAND_SWING;  // go to the other
  } else if ((a.gOffWhenDone || (a.isHand ? HAND_RELAX : GRIP_RELAX)) && now - a.gDone > 400) {
    a.gripper.detach();                                // arrived: servo OFF, no more twitching
    if (a.gOffWhenDone) { a.gOffWhenDone = false; tag(a); Serial.println(a.isHand ? F("hand servo OFF") : F("gripper servo OFF")); }
  }
}

// ============================ POSITIONS ============================
// Returns false if that position is not set
bool stepperToPos(Arm &a, int n) {
  if (n < 0 || n > 4 || a.mPos[n] == NONE) {
    tag(a); Serial.print(F("stepper: position ")); Serial.print(n); Serial.println(F(" is not set."));
    return false;
  }
  tag(a); Serial.print(F("stepper to position ")); Serial.println(n);
  stepperGoTo(a, a.mPos[n]);
  return true;
}

bool elbowToPos(Arm &a, int n) {
  if (n < 0 || n > 4 || a.sPos[n] == NONE) {
    tag(a); Serial.print(F("elbow: position ")); Serial.print(n); Serial.println(F(" is not set."));
    return false;
  }
  tag(a); Serial.print(F("elbow to position ")); Serial.println(n);
  servoGoTo(a, a.sPos[n]);
  return true;
}

// Name of the position a joint is at right now ("-" if between positions)
void printPosName(const float *pos, float angle, float tol) {
  for (int i = 0; i < 5; i++) {
    if (pos[i] != NONE && fabs(angle - pos[i]) <= tol) {
      Serial.print(F("  [at position ")); Serial.print(i); Serial.print(F("]"));
      return;
    }
  }
  Serial.print(F("  [between positions]"));
}

void printTableRow(const __FlashStringHelper *label, const float *pos, float lo, float hi) {
  Serial.print(label);
  for (int i = 0; i < 5; i++) {
    Serial.print(F("\t"));
    if (pos[i] == NONE) Serial.print(F("-")); else Serial.print((int)pos[i]);
  }
  Serial.print(F("\tlimits ")); Serial.print((int)lo); Serial.print(F("-")); Serial.println((int)hi);
}

void printTable() {
  Serial.println(F("POSITIONS (pot readings)  zero\t1\t2\t3\t4"));
  printTableRow(F("RIGHT stepper           "), R_M_POS, R_M_MIN, R_M_MAX);
  printTableRow(F("RIGHT elbow (servo deg) "), R_S_ANG, 0, 180);
  printTableRow(F("LEFT  stepper           "), L_M_POS, L_M_MIN, L_M_MAX);
  printTableRow(F("LEFT  elbow             "), L_S_POS, L_S_MIN, L_S_MAX);
}

// ============================ STATUS / HELP ============================
void printHelp() {
  Serial.println(F("--------------------------------------------------------------"));
  Serial.println(F("Arm:       r = RIGHT (default)   l = LEFT   b = BOTH"));
  Serial.println(F("Whole arm: 0..4         e.g. 2, l 0, b 1   (0 = zero position)"));
  Serial.println(F("Stepper:   m 0..4       fine: m r <deg> | m a <pot>   m v | m c | m x"));
  Serial.println(F("Jog:       m f = forward, m b = backward, until 'm x' or 'x'  (l m f = left)"));
  Serial.println(F("Elbow:     s 0..4       fine: s r <deg> | s a <pot>   s v | s x | s c"));
  Serial.println(F("Hand:      g s = shake | g n = no shake, then OFF   (right arm)"));
  Serial.println(F("Gripper:   l g o | l g c   (left arm)      g a <angle> = tuning"));
  Serial.println(F("Servo OFF: g f = hand OFF | l g f = gripper OFF | b g f = both OFF"));
  Serial.println(F("Right elbow: no pot, s 0..4 by timing, 0 = rest down"));
  Serial.println(F("Other:     p (status) | t (positions table) | x (EMERGENCY STOP) | h"));
  Serial.println(F("--------------------------------------------------------------"));
}

void printStatus(Arm &a) {
  tag(a); Serial.print(F("stepper: ")); Serial.print(a.mAngle);
  Serial.print(F("  target ")); Serial.print(a.mTarget);
  printPosName(a.mPos, a.mAngle, 3.0);
  Serial.println(a.mFault ? F("  [STOPPED]") : (a.mJog ? (a.mJogDir == backDir(a) ? F("  [jogging backward]") : F("  [jogging forward]")) : (a.mMoving ? F("  [moving]") : F(""))));

  tag(a); Serial.print(F("elbow:   "));
  if (a.sNoPot) {
    Serial.print(F("servo ")); Serial.print(a.sCmd); Serial.print(F(" deg (no pot)"));
    printPosName(a.sPos, a.sCmd, 1.0);
    Serial.println(!a.servo.attached() ? F("  [servo OFF]") : (a.sMoving ? F("  [moving]") : F("")));
  } else {
    Serial.print(a.sAngle);
    Serial.print(F("  target ")); Serial.print(a.sTarget);
    printPosName(a.sPos, a.sAngle, 3.0);
    Serial.print(a.sCalOK ? F("  feedback ON") : F("  feedback OFF"));
    Serial.println(a.sMoving && !a.sQuiet ? F("  [moving]") : F(""));
  }

  if (a.isHand) {
    tag(a); Serial.print(F("hand:    "));
    if (!a.gStarted) Serial.println(F("not started yet (send g s / g n)"));
    else { Serial.print(a.shaking ? F("SHAKING") : F("not shaking"));
           Serial.println(a.gripper.attached() ? F("") : F("  [servo OFF]")); }
  } else {
    tag(a); Serial.print(F("gripper: "));
    if (!a.gStarted) Serial.println(F("not started yet (send g o / g c)"));
    else { Serial.print(a.gripOpen ? F("OPEN") : F("CLOSED"));
           Serial.println(a.gripper.attached() ? F("") : F("  [servo OFF]")); }
  }
}

void emergencyStop() {
  Arm *arms[2] = { &armR, &armL };
  for (int i = 0; i < 2; i++) {
    stepperHold(*arms[i]); servoHold(*arms[i]);
    arms[i]->gGoal = arms[i]->gCmd;                    // gripper / hand stops where it is
    arms[i]->shaking = false;
  }
  Serial.println(F("!! EMERGENCY STOP: all joints holding where they are."));
}

// ============================ COMMANDS ============================
char *skipSpaces(char *p) { while (*p == ' ') p++; return p; }
bool isDigitC(char c) { return c >= '0' && c <= '9'; }

// Runs one command (without the arm letter) on one arm
void armCommand(Arm &a, char *cmd) {
  cmd = skipSpaces(cmd);
  char motor = cmd[0];
  if (motor == 'p') { printStatus(a); return; }

  // "2" -> whole arm (stepper + elbow) to position 2
  if (isDigitC(motor)) {
    int n = atoi(cmd);
    stepperToPos(a, n);
    elbowToPos(a, n);                                  // n = 0 on the right: elbow rests, then servo OFF
    if (n == 0 && a.isHand && a.shaking) handStill(a); // zero: the hand stops shaking
    return;
  }

  char *rest = skipSpaces(cmd + 1);
  char op = rest[0];
  float val = atof(rest + 1);

  // "m 3" / "s 1" -> joint to position
  if (isDigitC(op)) {
    int n = atoi(rest);
    if (motor == 'm') stepperToPos(a, n);
    else if (motor == 's') elbowToPos(a, n);
    else printHelp();
    return;
  }

  if (motor == 'm') {
    switch (op) {
      case 'a': stepperGoTo(a, val); break;                 // absolute pot reading
      case 'r': stepperGoTo(a, a.mTarget + val); break;     // relative degrees
      case 'v':
        if (val > 0) a.mSpeedDeg = val;
        tag(a); Serial.print(F("stepper speed = ")); Serial.println(a.mSpeedDeg);
        break;
      case 'c':
        if (val > 0) a.mAccelDeg = val;
        tag(a); Serial.print(F("stepper accel = ")); Serial.println(a.mAccelDeg);
        break;
      case 'x':
        stepperHold(a);
        tag(a); Serial.println(F("stepper stop, holding here"));
        break;
      case 'f': stepperJog(a, true);  break;                // rotate forward until stopped
      case 'b': stepperJog(a, false); break;                // rotate backward until stopped
      default: printHelp();
    }
  } else if (motor == 's') {
    switch (op) {
      case 'a': servoGoTo(a, val); break;                   // absolute pot reading
      case 'r': servoGoTo(a, a.sTarget + val); break;       // relative degrees
      case 'v':
        if (val > 0) a.sSpeedDeg = val;
        tag(a); Serial.print(F("elbow speed = ")); Serial.println(a.sSpeedDeg);
        break;
      case 'x':
        servoHold(a);
        tag(a); Serial.println(F("elbow stop, holding here"));
        break;
      case 'c':
        if (a.sNoPot) { tag(a); Serial.println(F("elbow has no pot: nothing to calibrate (positions are servo angles).")); break; }
        calibrateElbow(a);
        a.sLast = a.sSettleStart = millis();
        break;
      default: printHelp();
    }
  } else if (motor == 'g') {
    switch (op) {
      case 'o':
      case 'c':
        if (a.isHand) { tag(a); Serial.println(F("has a HAND: use 'g s' (shake) or 'g n' (no shake)")); }
        else if (op == 'o') gripperOpen(a);
        else gripperClose(a);
        break;
      case 's':
      case 'n':
        if (!a.isHand) { tag(a); Serial.println(F("has a GRIPPER: use 'g o' (open) or 'g c' (close)")); }
        else if (op == 's') handShake(a);
        else handStill(a);
        break;
      case 'f': gripperOff(a); break;                       // servo OFF right now
      case 'a': {                                           // tuning: gripper / hand to any angle
        int deg = constrain((int)val, 0, 180);
        gripperStart(a, deg);
        a.shaking = false;
        a.gGoal = deg;
        tag(a); Serial.print(F("gripper -> ")); Serial.print(deg);
        Serial.println(F(" deg (tuning). Put the good numbers in *_GRIP_OPEN / *_GRIP_CLOSE."));
        break;
      }
      default: printHelp();
    }
  } else {
    printHelp();
  }
}

void handleCommand(char *cmd) {
  cmd = skipSpaces(cmd);
  char first = cmd[0];
  bool alone = (*skipSpaces(cmd + 1) == '\0');

  if (first == 'x' && alone) { emergencyStop(); return; }
  if (first == 'h' || first == '?') { printHelp(); return; }
  if (first == 't' && alone) { printTable(); return; }
  if (first == 'p' && alone) { printStatus(armR); printStatus(armL); return; }

  if (first == 'l')      armCommand(armL, cmd + 1);
  else if (first == 'r') armCommand(armR, cmd + 1);
  else if (first == 'b') {                               // both arms
    char copy[32];
    strncpy(copy, cmd + 1, sizeof(copy) - 1); copy[sizeof(copy) - 1] = '\0';
    armCommand(armR, cmd + 1);
    armCommand(armL, copy);
  }
  else armCommand(armR, cmd);                            // no letter = RIGHT
}

char buf[32];
byte bufLen = 0;

void readSerial() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (bufLen > 0) { buf[bufLen] = '\0'; handleCommand(buf); bufLen = 0; }
    } else if (bufLen < sizeof(buf) - 1) {
      buf[bufLen++] = tolower(ch);
    }
  }
}

// ============================ MAIN ============================
void setupStepper(Arm &a) {
  a.stepper->setPinsInverted(true, true, false);   // common-anode wiring
  a.stepper->setMinPulseWidth(5);                  // DM542 needs >= 2.5 us
  a.stepper->setMaxSpeed(4000);                    // real limit is mSpeedDeg
}

void setup() {
  Serial.begin(115200);
  Serial.println(F("Starting both arms..."));

  initArm(armR, "RIGHT", &stepR, R_M_POT, R_M_POT_INVERT, R_M_DIR_INVERT, R_M_MIN, R_M_MAX, R_M_POS,
          R_SERVO_PIN, R_S_POT, 0, 180, R_S_ANG, R_GRIP_PIN, R_GRIP_OPEN, R_GRIP_CLOSE);
  armR.sNoPot = true;                              // right elbow: no pot, positions by timing
  armR.isHand = R_HAS_HAND;                        // right arm: shaking hand instead of gripper
  initArm(armL, "LEFT",  &stepL, L_M_POT, L_M_POT_INVERT, L_M_DIR_INVERT, L_M_MIN, L_M_MAX, L_M_POS,
          L_SERVO_PIN, L_S_POT, L_S_MIN, L_S_MAX, L_S_POS, L_GRIP_PIN, L_GRIP_OPEN, L_GRIP_CLOSE);

  setupStepper(armR);
  setupStepper(armL);

  startElbow(armR);                                // no jump with saved calibration
  startElbow(armL);

  unsigned long now = millis();
  Arm *arms[2] = { &armR, &armL };
  for (int i = 0; i < 2; i++) {
    Arm &a = *arms[i];
    a.mAngle = readPotDeg(a.mPotPin, a.mPotInvert);   // steppers hold where they are
    a.mTarget = a.mAngle;
    a.mLast = a.sLast = a.sSettleStart = now;
  }

  Serial.println(F("Both arms ready."));
  printTable();
  printStatus(armR);
  printStatus(armL);
  printHelp();
}

void loop() {
  stepR.runSpeed();     // call often: no delay() anywhere in loop
  stepL.runSpeed();
  readSerial();
  updateStepper(armR);
  updateStepper(armL);
  stepR.runSpeed();
  stepL.runSpeed();
  updateServo(armR);
  updateServo(armL);
  updateGripper(armR);
  updateGripper(armL);
}
