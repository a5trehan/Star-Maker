#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import sys
import time
from time import sleep
import argparse
import subprocess
subprocess.run(['sudo', 'pigpiod'])
import blobconverter
import numpy as np
import RPi.GPIO as GPIO
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# User configurable parameters
FRAME_SIZE = (300, 300) # Specify frame size (must match NN output size)
SHAVES = 5 # Specify vector processors for device
SPATIAL_CONFIDENCE = 0.8 # Specify confidence threshold for filtering spatial detection
DISPARITY_CONFIDENCE = 255 # Specify confidence threshold for disparity calculation
BB_SCALE = 0.5 # Specify bounding box scale factor
MIN_DEPTH = 100 # Specify minimum depth range (in millimeters)
MAX_DEPTH = 5500 # Specify maximum depth range (in millimeters)

# Assign pin numbers
PWR1, ENA1, IN1, IN2, GND = 2, 33, 31, 29, 39
PWR2, ENA2, IN3, IN4, GND = 4, 32, 18, 16, 34
GPIO.setmode(GPIO.BOARD)

# Compile and download mobilenet-ssd blob
nnPathDefault = blobconverter.from_zoo(
        name="mobilenet-ssd",
        shaves=SHAVES,
        zoo_type="intel"
    )

# Parse arguments for command-line interfacing
parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
parser.add_argument('-ff', '--full_frame', action="store_true", help="Perform tracking on full RGB frame", default=False)
args = parser.parse_args()
fullFrameTracking = args.full_frame

# Create pipeline
pipeline = dai.Pipeline()

# Define camera sources
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

# Instantiate relevant nodes
spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
stereo = pipeline.create(dai.node.StereoDepth)
objectTracker = pipeline.create(dai.node.ObjectTracker)

# Define outputs and set stream names
xoutRgb = pipeline.create(dai.node.XLinkOut)
trackerOut = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("preview")
trackerOut.setStreamName("tracklets")

# Configure RGB camera properties
camRgb.setPreviewSize(FRAME_SIZE[0], FRAME_SIZE[1])
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setInterleaved(False) # Store color channels in separate planes
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Configure grayscale camera properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Configure stereo node properties
stereo.initialConfig.setConfidenceThreshold(DISPARITY_CONFIDENCE)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB) # Align depth map to the perspective of RGB camera inferencing
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

# Configure spatial detection network properties
spatialDetectionNetwork.setBlobPath(args.nnPath)
spatialDetectionNetwork.setConfidenceThreshold(SPATIAL_CONFIDENCE)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(BB_SCALE)
spatialDetectionNetwork.setDepthLowerThreshold(MIN_DEPTH)
spatialDetectionNetwork.setDepthUpperThreshold(MAX_DEPTH)

# Configure object tracker properties
objectTracker.setDetectionLabelsToTrack([15])  # Limit mobilenet-ssd tracking to only people
# Tracker Types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
# ID Assignment Policies: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID) # Obtain the smallest ID when a new object is tracked

# Link mono camera outputs to stereo node
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Link RGB camera output to spatial detection network input
camRgb.preview.link(spatialDetectionNetwork.input)

# Link passthrough images to preview stream (for non-blocking queue)
objectTracker.passthroughTrackerFrame.link(xoutRgb.input)

# Link object tracking results to tracklet stream
objectTracker.out.link(trackerOut.input)

if fullFrameTracking:
    # Adjust aspect ratio after downscaling from video size
    camRgb.setPreviewKeepAspectRatio(False)
    # Link RGB
    camRgb.video.link(objectTracker.inputTrackerFrame)
    objectTracker.inputTrackerFrame.setBlocking(False)
    # do not block the pipeline if it's too slow on full frame
    objectTracker.inputTrackerFrame.setQueueSize(2)
else:
    spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
spatialDetectionNetwork.out.link(objectTracker.inputDetections)
stereo.depth.link(spatialDetectionNetwork.inputDepth)

def motor_setup():
    # Setup base servo - choose a GPIO pin for servo input
    factory = PiGPIOFactory() 
    servo = AngularServo(21, min_angle=0, max_angle=270, min_pulse_width=0.0005, max_pulse_width=0.0025, pin_factory=factory)
    servo.angle = 140

    # Setup launch servo setup - choose a GPIO pin for servo input 
    servoR = AngularServo(27, min_angle=0, max_angle=180, min_pulse_width=0.0009, max_pulse_width=0.0021, pin_factory=factory)
    servoL = AngularServo(17, min_angle=0, max_angle=180, min_pulse_width=0.0009, max_pulse_width=0.0021, pin_factory=factory)
    # configured for 130 degrees actuation 
    servoR.angle = 130
    servoL.angle = 0

    # Initialize PWM
    PWMA = gpio_setup(IN1, IN2, ENA1)
    PWMB = gpio_setup(IN3, IN4, ENA2)
    PWMA.start(0)
    PWMB.start(0)
    #PWMA.ChangeFrequncy

    # Initialize direction of operation 
    GPIO.output(IN1,GPIO.HIGH) #CW
    GPIO.output(IN2,GPIO.LOW)
    GPIO.output(IN3,GPIO.HIGH) #CW
    GPIO.output(IN4,GPIO.LOW)

    return servo, servoR, servoL, PWMA, PWMB

def gpio_cleanup():
    PWMA.stop()
    PWMB.stop()
    GPIO.cleanup()

def gpio_setup(CW, CCW, ENA):
    GPIO.setup(ENA, GPIO.OUT)
    GPIO.setup(CW, GPIO.OUT)
    GPIO.setup(CCW, GPIO.OUT) 
    return GPIO.PWM(ENA, 1000) # (channel, frequency)

def drop_ball():
    # configured for 130 degrees actuation & 2 sec interval between operation
    #sleep(0.5)
    servoR.angle = 0
    servoL.angle = 130
    sleep(0.5)
    servoR.angle = 130
    servoL.angle = 0
    #sleep(0.5)

def launch_ball(time, depth):
    max_speed = 4.1904*np.exp(depth*0.0008)
    speed = min(80, max_speed)
    try:
        # 3s acceleration
        for dc in np.linspace(0,speed,num=20):
            PWMA.ChangeDutyCycle(dc)
            PWMB.ChangeDutyCycle(dc)
            sleep(0.1)
        drop_ball()
        sleep(time)
        # 3s deceleration
        for dc in np.linspace(speed,0,num=20):
            PWMA.ChangeDutyCycle(dc)
            PWMB.ChangeDutyCycle(dc)
            sleep(0.1)
    except KeyboardInterrupt:
        gpio_cleanup()

# Set base servo angle with speed of (angle_step / time_step) in deg/s
def orient_base(des_angle):
    time_step = 0.02
    angle_step = 0.5
    curr_angle = servo.angle
    if curr_angle < des_angle:
        dir = 1 # CW
    else:
        dir = -1 # CCW
    while abs(curr_angle - des_angle) > angle_step:
        curr_angle += dir*angle_step
        servo.angle = curr_angle
        sleep(time_step)
    servo.angle = des_angle

def exit_process():
    print("Exiting...")
    gpio_cleanup()
    orient_base(140)
    cv2.destroyAllWindows()
    sys.exit()

# Upload the pipeline to the device
with dai.Device(pipeline) as device:

    preview = device.getOutputQueue("preview", 4, False)
    tracklets = device.getOutputQueue("tracklets", 4, False)

    start_time = time.monotonic()
    prev_time = start_time
    counter = 0
    fps = 0
    color = (255, 255, 255)
    prev_dir = 0
    angle_step = 0.4
    launch = False
    launching = False
    launched = False
    launch_count = 0
    last_launch_time = 0

    # Outline the exercise program
    launch_period = 10 # in seconds
    launch_duration = 2 # in seconds
    num_launches = 1

    # Initialize all motor objects
    servo, servoR, servoL, PWMA, PWMB = motor_setup()
 
    while(True):
        imgFrame = preview.get()
        track = tracklets.get()

        counter+=1
        current_time = time.monotonic()
        elapsed_time = current_time - start_time

        # Signal the projectile launch instruction
        if (elapsed_time - last_launch_time) >= launch_period and launch_count < num_launches:
            launch = True

        if (current_time - prev_time) > 1 :
            fps = counter / (current_time - prev_time)
            counter = 0
            prev_time = current_time

        frame = imgFrame.getCvFrame()
        trackletsData = track.tracklets
        tracklets_length = len(trackletsData)
        for t in trackletsData:
            # Dynamically set deviation tolerance
            deviation_tol = round((1/80)*t.spatialCoordinates.z + (35/4))
            
            if launch:
                # Call launch process
                launch_ball(launch_duration, t.spatialCoordinates.z)
                launch_count += 1
                last_launch_time = time.monotonic() - start_time
                launch = False

            if t.status.name == "TRACKED" and tracklets_length == 1 and launch == False:
                curr_dir = -np.sign(t.spatialCoordinates.x)
                #if abs(t.spatialCoordinates.x) > deviation_tol:
                correction_angle = np.rad2deg(np.arctan(abs(t.spatialCoordinates.x)/t.spatialCoordinates.z))
                #print(correction_angle)
                #print(t.spatialCoordinates.z)
                if correction_angle > angle_step:
                    servo.angle += curr_dir*angle_step
                prev_dir = curr_dir
             
            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            try:
                label = labelMap[t.label]
            except:
                label = t.label

            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

            cv2.putText(frame, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        cv2.imshow("Star-Maker Preview", frame)

        if cv2.waitKey(1) == 27:
            break
    
    sleep(1)
    exit_process()