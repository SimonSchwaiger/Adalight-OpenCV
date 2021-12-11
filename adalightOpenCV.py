#!/env/bin/python

from os import replace
import cv2
from PIL import Image

import serial
from serial import Serial

import numpy as np

import time

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
        # Set resolution
        #self.capture.set(cv2.CV_CAP_PROP_FRAME_WIDTH, 640)
        #self.capture.set(cv2.CV_CAP_PROP_FRAME_WIDTH, 480)
        # Open capture
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
        # Store range of tv in mask
        self.range = None
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
    def readFrame(self, scalingFactor=1, debug=False):
        # Get timestamp to measure how long the method takes in debug mode
        if debug: stamp = time.time()
        # Read frame
        ret, frame = self.capture.read()
        # TODO check return value
        # Scale image if scalingFactor is not 1
        # This is useful if you have a high res camera and want to improve performance
        if scalingFactor != 1: frame = self.scaleFrame(frame, scalingFactor)
        self.latestFrame = frame
        # If this is the first image, store as reference
        if self.weight == 1:
            self.reference = frame
            self.updateWeight()
        else:
            # Update reference image
            self.reference = self.reference*(1-self.weight) + frame*self.weight
            self.updateWeight()
        # Print processing time
        if debug: print("readFrame processing time: {}".format(time.time() - stamp))
    #
    def updateMask(self, threshhold=70):
        # Get thressholded reference image and inverted version
        reference = cv2.cvtColor(self.reference.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        _, reference = cv2.threshold(reference,threshhold,255,cv2.THRESH_BINARY)
        #printImage(reference)
        inv = cv2.bitwise_not(reference)
        # Get image shape
        height, width = reference.shape
        # Floodfill reference image from the middle of the image and combine both masks to get tv shape
        cv2.floodFill(reference,None,(int(height/2),int(width/2)),0)
        mask = cv2.bitwise_not(reference|inv)
        #
        # Get and store the range of rows and cols in the mask
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
        # Store the range of rows and cols as class member
        self.range = [ firstRow, lastRow, firstCol, lastCol ]
        #
        # Setup replacement array for cropping out TV
        replacement = np.array([ [127, 127, 127] for i in range(len(mask)*len(mask[0])) ]).reshape(len(mask), len(mask[0]), 3)
        # Only colour outside of TV in replacement
        replacement = cv2.bitwise_and(replacement,replacement,mask=cv2.bitwise_not(mask))
        self.replacement = replacement.astype('uint8')
        #
        # Store mask
        self.mask = mask
    #
    def getLatestFrame(self):
        return self.latestFrame
    #
    def getLEDColours(self, frame, numLEDs=60, ledHeight=1 ):
        # Replace all potentially remaining background pixel with gray
        # This is done to reduce their influence when downscaling the cropped image to get LED values
        # This approach is really terrible for accuracy, but the best thing i could come up with without requiring complicated transforms
        #TODO: figure out how to color the background like screen edges (change creation of the replacement array)
        # Set up replacement as gray in updateMask() Method
        # Only colour inside of TV in replacement
        frame = cv2.bitwise_and(frame, frame, mask=self.mask)
        # Combine both images together
        frame = cv2.bitwise_or(frame, self.replacement)
        #
        # Crop to only include TV
        firstRow, lastRow, firstCol, lastCol = self.range
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

        Sources for communication protocol:
        https://forums.adafruit.com/viewtopic.php?f=47&t=29970
        https://github.com/adafruit/Adalight/blob/master/Processing/Adalight/Adalight.pde starting at line 247

        """
        # Adjust LED brightness for white to not overpower everything else
        # All values are normalised to a combined sum of 255
        scale = np.array([ max(np.sum(entry)/255, 1.0) for entry in data ])
        data = np.array([ entry/scale[i] for i, entry in enumerate(data) ])
        #data = np.clip(data, 0, 255)        # for safety (TODO: is clipping really needed?)
        data = data.astype(np.uint8)
        # Start format data as ada bytearray
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
        # Append R, G and B data for each led
        # OpenCV format: BGR
        # Adalight format: GRB
        for entry in data:
            packet.extend(bytes([entry[1]]))
            packet.extend(bytes([entry[2]]))
            packet.extend(bytes([entry[0]]))
        # Send packet and return
        return self.ser.write(packet)

# Kelvin table for colour warmth
# http://www.vendian.org/mncharity/dir3/blackbody/
kelvinTable = {
    1000: (255,56,0),
    1500: (255,109,0),
    2000: (255,137,18),
    2500: (255,161,72),
    3000: (255,180,107),
    3500: (255,196,137),
    4000: (255,209,163),
    4500: (255,219,186),
    5000: (255,228,206),
    5500: (255,236,224),
    6000: (255,243,239),
    6500: (255,249,253),
    7000: (245,243,255),
    7500: (235,238,255),
    8000: (227,233,255),
    8500: (220,229,255),
    9000: (214,225,255),
    9500: (208,222,255),
    10000: (204,219,255)}

class colourTransformer():
    """ 
    Helper class to provide colour correction to processed images.

    Source for each method is in the comments of the respective method.
    """
    def __init__(self, config) -> None:
        # Get config params
        self.gamma = config.get('Gamma')
        #self.saturation = config.get('Saturation')
        #self.colourTemp = config.get('ColourTemp')
        #self.autoWhitebalance = config.get('AutoWhitebalance')
        # Set gamma and lookup table based on gamma
        inv = 1/self.gamma
        self.gammaLUT = np.array([ ((i/255.0)**inv)*255 for i in np.arange(0, 256) ]).astype("uint8")
        # Set placeholder and wight for motion smoothing
        self.lastFrame = None
        self.smoothingWeight = config.get('SmoothingWeight')
    #
    def applyGammaCorrection(self, frame):
        return cv2.LUT(frame, self.gammaLUT)
    #
    def applySaturation(self, frame, saturation): #TODO: fix invalid number of channels error
        """ Changes the image saturation"""
        # https://answers.opencv.org/question/193336/how-to-make-an-image-more-vibrant-in-colour-using-opencv/
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame = frame.astype(np.float64)
        frame[...,1] *= saturation
        frame = frame.astype(np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    #
    def applyWhitebalance(self, frame):
        """ Applies automatic whitebalance to the image """
        # https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    #
    def applyColourTemp(self, frame, temp):
        """ Changes the colour temperature of the image """
        # https://stackoverflow.com/questions/11884544/setting-color-temperature-for-a-given-image-like-in-photoshop
        # Get linear scale for each channel
        r, g, b = kelvinTable[temp]
        # Transform each channel in a linear fashion (OpenCV images are BGR)
        frame = frame.astype(np.float64)
        frame[:,:,0] *= b/255.0
        frame[:,:,1] *= g/255.0
        frame[:,:,2] *= r/255.0
        return frame.astype(np.uint8)
    #
    def applyMotionSmoothing(self, frame):
        """ Smooth compared to last frame """
        if np.all(self.lastFrame != None):
            weight = self.smoothingWeight
            frame = frame*weight + self.lastFrame*(1-weight)
            frame = frame.astype(np.uint8)
        # Store as new latest frame
        self.lastFrame = frame
        # Return processed frame
        return frame

if __name__=="__main__":
    #############################################
    #####              CONFIG               #####

    # Check webcams:
    # v4l2-ctl --list-devices

    # Check Serial Port
    # sudo dmesg | grep tty

    ## Config params
    numLEDs = 79        # Number of LEDs per row
    baudr = 115200
    portn = "/dev/ttyUSB0"
    cameraName = "/dev/video2"

    imageconfig = {
        'ColourTemp': None,
        'Autowhitebalance': None,
        'Gamma': 0.7,
        'Saturation': 1.3,
        'SmoothingWeight': 0.4
    }

    frameCount = 0
    maskUpdateRate = 200
    debug = True
    measureProcessingTime = True
    testSerial = False

    #############################################
    #####         DEBUG APPLICATION         #####

    # Instantiate serial communication and image reader
    if testSerial: ser = adalightServer(portn=portn, baudr=baudr)
    reader = imageReader(cameraName=cameraName)
    transformer = colourTransformer(imageconfig)

    # Read frames to get reference
    for i in range(400): reader.readFrame()

    reader.updateMask()

    # Get mask for debugging
    if debug: printImage(reader.reference)
    if debug: printImage(reader.mask)

    print("Setup Done. Starting infinite Loop.")

    while True:
        # Create timestamp for performance measurements
        if measureProcessingTime: stamp = time.time()
        # Read frame
        reader.readFrame(debug=debug)
        frame = reader.getLatestFrame()
        # Get processing time
        if measureProcessingTime: print("readFrame time: {}".format(time.time() - stamp))
        if measureProcessingTime: stamp = time.time()
        if debug: printImage(frame)
        # Automatic whitebalance
        if imageconfig.get('AutoWhitebalance') != None: 
            frame = transformer.applyWhitebalance(frame)
        if measureProcessingTime: print("Whitebalance time: {}".format(time.time() - stamp))
        if measureProcessingTime: stamp = time.time()
        if debug: printImage(frame)
        # Alter colour temperature
        if imageconfig.get('ColourTemp') != None:
            frame = transformer.applyColourTemp()
        if measureProcessingTime: print("Colourtemp time: {}".format(time.time() - stamp))
        if measureProcessingTime: stamp = time.time()
        if debug: printImage(frame)
        # Gamma correction
        if imageconfig.get('Gamma') != 1:
            frame = transformer.applyGammaCorrection(frame)
        if measureProcessingTime: print("Gamma processing time: {}".format(time.time() - stamp))
        if measureProcessingTime: stamp = time.time()
        if debug: printImage(frame)
        # Alter saturation
        if imageconfig.get('Saturation') != 1:
            frame = transformer.applySaturation(frame, imageconfig.get('Saturation'))
        if measureProcessingTime: print("Saturation processing time: {}".format(time.time() - stamp))
        if measureProcessingTime: stamp = time.time()
        if debug: printImage(frame)
        # Get LED colours
        data = reader.getLEDColours(frame, numLEDs=numLEDs)
        if measureProcessingTime: print("getLEDColours processing time: {}".format(time.time() - stamp))
        if measureProcessingTime: stamp = time.time()
        if debug: printImage(data)
        # Apply motion smoothing
        data = transformer.applyMotionSmoothing(data)
        if measureProcessingTime: print("Motion Smoothing processing time: {}".format(time.time() - stamp))
        if measureProcessingTime: stamp = time.time()
        # Send LED data to arduino
        if testSerial: ser.sendLEDData(numLEDs=numLEDs, data=data[0])
        # Update mask in slower intervals to save performance
        if frameCount == maskUpdateRate:
            reader.updateMask
            frameCount = 0
        else:
            frameCount += 1
        # Only do loop once in debug mode
        if debug | measureProcessingTime: break

    # Turn off
    if testSerial: ser.sendLEDData(numLEDs=79, data = [[0, 0, 0] for i in range(79)])


# This on arduino
#https://github.com/dmadison/Adalight-FastLED






