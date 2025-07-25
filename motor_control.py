import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import threading

# Lock to prevent simultaneous servo access from multiple threads
servo_lock = threading.Lock()

# Setup I2C communication using the board's default SCL and SDA pins
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize the PCA9685 PWM controller over I2C
pca = PCA9685(i2c)

# Set the frequency to 50Hz, appropriate for servos
pca.frequency = 50

# Create Servo objects for channels 0–5 with specific pulse width ranges
servo0 = servo.Servo(pca.channels[0], min_pulse=600, max_pulse=2300)
servo1 = servo.Servo(pca.channels[1], min_pulse=600, max_pulse=2300)
servo2 = servo.Servo(pca.channels[2], min_pulse=600, max_pulse=2300)
servo3 = servo.Servo(pca.channels[3], min_pulse=600, max_pulse=2300)
servo4 = servo.Servo(pca.channels[4], min_pulse=600, max_pulse=2300)
servo5 = servo.Servo(pca.channels[5], min_pulse=600, max_pulse=2300)

# Function to set the servos to a default "resting" or starting position
def default_position():
    # Base rotation - 90° points forward
    servo0.angle = 90

    # Shoulder servo - 0° is down/rest, increasing lifts the arm
    servo1.angle = 0

    # Elbow servo - 180° is default/neutral
    servo2.angle = 180

    # Wrist pitch - 180° is neutral
    servo3.angle = 180

    # Wrist roll - 0° is resting, 180° would rotate it fully
    servo4.angle = 0

    # Gripper - 180° is closed, 80° is max open
    servo5.angle = 180

# Function to move the robot's arm to a specific bin based on the color
def identify_bin(s):
    if s == "Red":
        time.sleep(1)
        servo0.angle = 180       # Rotate base to Red bin
        servo1.angle = 100       # Lower arm toward bin
        time.sleep(1)
        servo5.angle = 150       # Release gripper slightly
        time.sleep(1)
    elif s == "Yellow":
        time.sleep(1)
        servo0.angle = 135       # Rotate to Yellow bin
        servo1.angle = 100
        time.sleep(1)
        servo5.angle = 150
        time.sleep(1)
    elif s == "Green":
        time.sleep(1)
        servo0.angle = 45        # Rotate to Green bin
        servo1.angle = 100
        time.sleep(1)
        servo5.angle = 150
        time.sleep(1)
    elif s == "Blue":
        time.sleep(1)
        servo0.angle = 0         # Rotate to Blue bin
        servo1.angle = 100
        time.sleep(1)
        servo5.angle = 150
        time.sleep(1)

# High-level function to perform the entire sorting operation for a given color
def color_sort(s):
    with servo_lock:
        # Move all servos to default position
        default_position()
        
        # Wait to ensure servos reach position
        time.sleep(7)

        # Simulate grabbing a cube
        servo5.angle = 150       # Start closing gripper slightly
        servo1.angle = 130       # Lower arm to pick position
        time.sleep(1)

        servo5.angle = 180       # Close gripper to grab the cube
        time.sleep(1)
        servo1.angle = 0         # Lift arm with cube
        
        # Move arm to correct bin and release cube
        identify_bin(s)

        # Return to default position after sorting
        default_position()
