#!/env/bin/python
import cv2
from PIL import Image
#from skimage.metrics import structural_similarity as ssim

import serial
from serial import Serial

import numpy as np

import time

# Check webcams:
# v4l2-ctl --list-devices

# Check Serial Port
# sudo dmesg | grep ttyy

def printImage(img):
    """ Print image on screen using Pillow for debugging """
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Image.fromarray(img).show()

class imageReader():
    """
    This implementation is probably not the most elegant way to to this.
    But since camera and TV should be static anyways, it should be fine.
    Furthermore, this way of detecting the TV automatically deals with 
    content with black bars.

    For TV detection, recorded frames are averaged. Over time, this should
    turn the screen in the reference image white. We then thresshold the
    reference image and fill from image centre (where the TV hopefully is),
    in order to get a mask, representing the TV's location in the frame.
    """
    def __init__(self, cameraName=" "):
        # Open Webcam 
        self.capture = cv2.VideoCapture()
        self.capture.open(cameraName)
        # Check if video catpture has been opened
        if not self.capture.isOpened():
            raise IOError("Cannot open webcam")
        # Init value for weight calculation of reference
        self.reference = None
        self.weight = 1
        # Store latest frame and mask that identifies the tv
        self.latestFrame = None
        self.mask = None
    #
    def updateWeight(self):
        tmp = 1/self.weight
        self.weight = 1/(tmp+1)
    #
    def scaleFrame(self, frame, scale):
        width = int(len(frame[0])/scale)
        height = int(len(frame)/scale)
        return cv2.resize(frame,(width, height), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
    #
    def readFrame(self, scalingFactor=2):
        # Read frame
        ret, frame = self.capture.read()
        # TODO check return value
        # Convert to greyscale and downscale for easier processing
        frameBw = self.scaleFrame(frame, scalingFactor)
        self.latestFrame = frameBw
        # If this is the first image, store as reference
        if self.weight == 1:
            self.reference = frameBw
            self.updateWeight()
        else:
            # Update reference image
            self.reference = self.reference*(1-self.weight) + frameBw*self.weight
            self.updateWeight()
    #
    def updateMask(self, threshhold=60):
        # Get thressholded reference image and inverted version
        reference = cv2.cvtColor(self.reference.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        _, reference = cv2.threshold(reference,threshhold,255,cv2.THRESH_BINARY)
        printImage(reference)
        inv = cv2.bitwise_not(reference)
        # Get image shape
        height, width = reference.shape
        # Floodfill reference image from the middle of the image and combine both masks to get tv shape
        cv2.floodFill(reference,None,(int(height/2),int(width/2)),0)
        self.mask = cv2.bitwise_not(reference|inv)
    #
    def getLEDColours(self, numLEDs=60, ledHeight=1 ):
        # Get number of rows
        mask = self.mask
        # Get first and last row containing TV
        tmp = [ np.any(mask[i]!=0) for i in range(len(mask)) ]
        tmp = np.where(np.array(tmp) == True)
        firstRow = tmp[0][0]
        lastRow = tmp[0][-1]
        #
        # Get first and last col containing TV
        tmp = [ np.any(mask[:,i]!=0) for i in range(len(mask[0])) ]
        tmp = np.where(np.array(tmp) == True)
        firstCol = tmp[0][0]
        lastCol = tmp[0][-1]
        #
        # Replace all potentially remaining background pixel with gray
        # This is done to reduce their influence when downscaling the cropped image to get LED values
        # This approach is really terrible for accuracy, but the best thing i could come up with without requiring complicated transforms
        #TODO: figure out how to color the background like screen edges (change creation of the replacement array)
        # Set up replacement as gray
        replacement = np.array([ [127, 127, 127] for i in range(len(mask)*len(mask[0])) ]).reshape(len(mask), len(mask[0]), 3)
        # Only colour outside of TV in replacement
        replacement = cv2.bitwise_and(replacement,replacement,mask=cv2.bitwise_not(mask))
        # Only colour inside of TV in replacement
        frame = cv2.bitwise_and(self.latestFrame, self.latestFrame,mask=mask)
        # Combine both images together
        frame = cv2.bitwise_or(frame, replacement.astype('uint8'))
        #
        # Crop to only include TV
        frame = frame[firstRow:lastRow][:,firstCol:lastCol]
        #
        # Resize to horizontal resolution == num of LEDs
        width = numLEDs
        height = ledHeight
        colours = cv2.resize(frame,(width, height), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        return colours

def byte_xor(ba1, ba2):
    """ 
    Performs element-wise XOR between two bytes. Source:
    https://nitratine.net/blog/post/xor-python-byte-strings/
    """
    return bytes([_a ^ _b for _a, _b in zip(ba1, ba2)])

class adalightServer():
    """ Connects to serial port and transmits LED data using the Adalight format """
    def __init__(self, portn="/dev/ttyUSB0", baudr=115200) -> None:
        """ Class Constructor: Initiates serial connection """
        # Init serial connection
        try:
            ##Serial connection to the Arduino
            self.ser = serial.Serial(
                port=portn ,\
                baudrate=baudr,\
                parity=serial.PARITY_NONE,\
                stopbits=serial.STOPBITS_ONE,\
                bytesize=serial.EIGHTBITS,\
                timeout=0)
            print("Connected to: " + self.ser.portstr)
        except (IOError, OSError):
            print ("Serial error" + "\n Exiting Serial Connection \n")
            #quit()
    #
    def __del__(self):
        """ Class Destructor: terminates serial connection """
        try:
            self.ser.close()
        except AttributeError:
            pass
    #
    def sendLEDData(self, data=[], numLEDs=0):
        """
        Sends LED data formatted in Ada format over serial.
        Data must be of length numLEDs and contain three channels going from 0-255

        The message format is defined like this:
        
        Byte #  Value
        0       'A' (0x41)
        1       'd' (0x64)
        2       'a' (0x61)
        3       LED count, high byte
        4       LED count, low byte
        5       Checksum (high byte XOR low byte XOR 0x55)

        For each LED:
        Byte #  Desc
        0       red (0-255)
        1       green (0-255)
        2       blue (0-255)

        Sources:
        https://forums.adafruit.com/viewtopic.php?f=47&t=29970
        https://github.com/adafruit/Adalight/blob/master/Processing/Adalight/Adalight.pde starting at line 247

        """
        packet = bytearray()
        # Add Ada and LED count
        packet.append(0x41)
        packet.append(0x64)
        packet.append(0x61)
        countBytes = (numLEDs).to_bytes(2, byteorder='big')
        packet.extend(countBytes)
        # Add checksum to the packet
        checkSum = byte_xor(bytes([countBytes[0]]),bytes([countBytes[1]]))
        checkSum = byte_xor(checkSum, bytes([0x55]))
        packet.extend(checkSum)
        # Append R, G and B data for each led Adalight Format = GRB
        # BGR
        # GRB
        for entry in data:
            packet.extend(bytes([entry[1]]))
            packet.extend(bytes([entry[2]]))
            packet.extend(bytes([entry[0]]))
        # Send packet and return
        return self.ser.write(packet)

## Config params
numLEDs = 79        # Number of LEDs per row
baudr = 115200
portn = "/dev/ttyUSB0"

framerate = 60

#############################################

reader = imageReader(cameraName="/dev/video2")

# Read frames to get reference
for i in range(500): reader.readFrame()

reader.updateMask()

# Get mask for debugging
#printImage(reader.reference)
#printImage(reader.mask)

# Instantiate serial communication and image reader
ser = adalightServer(portn=portn, baudr=baudr)



while True:
    reader.readFrame()
    data = reader.getLEDColours(numLEDs=numLEDs)
    #TODO colour correction
    # gamma & saturation
    ser.sendLEDData(numLEDs=79, data=data[0])
    time.sleep(1/framerate)



# Turn off
ser.sendLEDData(numLEDs=79, data = [[0, 0, 0] for i in range(79)])





# This on arduino
#https://github.com/dmadison/Adalight-FastLED


