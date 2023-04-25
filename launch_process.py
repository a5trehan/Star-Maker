#Imported Libraries
import RPi.GPIO as GPIO
from gpiozero import AngularServo
from time import sleep
import numpy as np
from gpiozero.pins.pigpio import PiGPIOFactory

# sudo pigiod - run in terminal intially
factory = PiGPIOFactory()

# Setup launch servo setup - choose a GPIO pin for servo input 
servoR = AngularServo(27, min_angle=0, max_angle=180, min_pulse_width=0.0009, max_pulse_width=0.0021, pin_factory=factory)
servoL = AngularServo(17, min_angle=0, max_angle=180, min_pulse_width=0.0009, max_pulse_width=0.0021, pin_factory=factory)
# configured for 130 degrees actuation 
r_angle = 130
servoR.angle = r_angle
servoL.angle = 0

pwm_freq = 1000 # in Hz
launch_depth = 2641.6 # in mm
launch_duration = 3 # in seconds

# Assign pin numbers
PWR1, ENA1, IN1, IN2, GND = 2, 33, 31, 29, 39
PWR2, ENA2, IN3, IN4, GND = 4, 32, 18, 16, 34
GPIO.setmode(GPIO.BOARD)

def cleanup(PWMA, PWMB):
    PWMA.stop()
    PWMB.stop()
    GPIO.cleanup()

def setup(CW, CCW, ENA):
    GPIO.setup(ENA, GPIO.OUT)
    GPIO.setup(CW, GPIO.OUT)
    GPIO.setup(CCW, GPIO.OUT) 
    return GPIO.PWM(ENA, pwm_freq) # (channel, frequency)

def drop_ball(r_angle, r_time):
    # configured for 130 degrees actuation & 2 sec interval between operation
    #sleep(r_time)
    servoR.angle = 0
    servoL.angle = r_angle
    sleep(r_time)
    servoR.angle = r_angle
    servoL.angle = 0
    #sleep(r_time)

def launch(PWMA, PWMB, time, depth):
    max_speed = 4.1904*np.exp(depth*0.0008)
    speed = min(80, max_speed)
    try:
        # 3s acceleration
        for dc in np.linspace(0,speed,num=30):
            PWMA.ChangeDutyCycle(dc)
            PWMB.ChangeDutyCycle(dc)
            sleep(0.1)
        drop_ball(130,0.5)
        sleep(time)
        # 3s deceleration
        for dc in np.linspace(speed,0,num=30):
            PWMA.ChangeDutyCycle(dc)
            PWMB.ChangeDutyCycle(dc)
            sleep(0.1)
    except KeyboardInterrupt:
        cleanup(PWMA, PWMB)

if __name__ == "__main__":
    # Initialize PWM
    PWMA = setup(IN1, IN2, ENA1)
    PWMB = setup(IN3, IN4, ENA2)
    PWMA.start(0)
    PWMB.start(0)
    #PWMA.ChangeFrequncy

    # Initialize direction of operation 
    GPIO.output(IN1,GPIO.HIGH) #CW
    GPIO.output(IN2,GPIO.LOW)
    GPIO.output(IN3,GPIO.HIGH) #CW
    GPIO.output(IN4,GPIO.LOW)

    # Call launch process
    launch(PWMA, PWMB, launch_duration, launch_depth)

    cleanup(PWMA, PWMB)