import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import threading
servo_lock = threading.Lock()

# Setup I2C and PCA9685
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

# Create Servo objects for each channel
servo0 = servo.Servo(pca.channels[0], min_pulse=600, max_pulse=2300)
servo1 = servo.Servo(pca.channels[1], min_pulse=600, max_pulse=2300)
servo2 = servo.Servo(pca.channels[2], min_pulse=600, max_pulse=2300)
servo3 = servo.Servo(pca.channels[3], min_pulse=600, max_pulse=2300)
servo4 = servo.Servo(pca.channels[4], min_pulse=600, max_pulse=2300)
servo5 = servo.Servo(pca.channels[5], min_pulse=600, max_pulse=2300)

# From bottom of robot to top:
# Default/Starting positions: 90, 0, 180, 180, 0, 180

def default_position():
    # 90 faces forwards, 0 faces left, 180 faces right
    servo0.angle = 90
    
    # 180 turns the arm too forward, set to 125 max
    # 0 is resting position
    # Increase bends Forward
    # Decrease Bends Backwards
    servo1.angle = 0
    
    # 180 is default/starting position
    servo2.angle = 180
    
    # 0 goes "backwards", 180 is "rest" position
    servo3.angle = 180
    
    # 0 is "resting position", 180 is upside down
    servo4.angle = 0
    
    # 80 is the farthest without "breaking" the robot, 180 closes the gripper
    servo5.angle = 180


def identify_bin(s):
    if s == "Red":
        time.sleep(1)
        servo0.angle = 180
        servo1.angle = 100
        time.sleep(1)
        servo5.angle = 150
        time.sleep(1)
    elif s == "Yellow":
        time.sleep(1)
        servo0.angle = 135
        servo1.angle = 100
        time.sleep(1)
        servo5.angle = 150
        time.sleep(1)
    elif s == "Green":
        time.sleep(1)
        servo0.angle = 45
        servo1.angle = 100
        time.sleep(1)
        servo5.angle = 150
        time.sleep(1)
    elif s == "Blue":
        time.sleep(1)
        servo0.angle = 0
        servo1.angle = 100
        time.sleep(1)
        servo5.angle = 150
        time.sleep(1)

def color_sort(s):
    # Set default position
    with servo_lock:
        default_position()
        
        # Give it a bit of time to adjust
        time.sleep(7)
        
        # Grabbing the cube
        servo5.angle = 150
        servo1.angle = 130
        
        time.sleep(1)
        
        servo5.angle = 180
        time.sleep(1)
        servo1.angle = 0
        
        
        # Determining the color and move cube
        identify_bin(s)
        
        default_position()