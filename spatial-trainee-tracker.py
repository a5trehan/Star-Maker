#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import time
import argparse
import blobconverter

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# User configurable parameters
FRAME_SIZE = (300, 300) # Specify frame size (must match NN output size)
SHAVES = 5 # Specify vector processors for device
SPATIAL_CONFIDENCE = 0.6 # Specify confidence threshold for filtering spatial detection
DISPARITY_CONFIDENCE = 255 # Specify confidence threshold for disparity calculation
BB_SCALE = 0.5 # Specify bounding box scale factor
MIN_DEPTH = 100 # Specify minimum depth range (in millimeters)
MAX_DEPTH = 5500 # Specify maximum depth range (in millimeters)

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
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
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

# Upload the pipeline to the device
with dai.Device(pipeline) as device:

    preview = device.getOutputQueue("preview", 4, False)
    tracklets = device.getOutputQueue("tracklets", 4, False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)

    while(True):
        imgFrame = preview.get()
        track = tracklets.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        frame = imgFrame.getCvFrame()
        trackletsData = track.tracklets
        for t in trackletsData:
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