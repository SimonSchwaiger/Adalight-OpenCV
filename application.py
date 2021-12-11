#!/env/bin/python

from adalightOpenCV import *
import json

def main():
    # Load config
    with open('./config.json') as json_file:
        config = json.load(json_file)
    # Instantiate Ambilight Components
    ser = adalightServer(portn=config.get('portn'), baudr=config.get('baudr'))
    reader = imageReader(cameraName=config.get('cameraName'))
    transformer = colourTransformer(config)
    frameCount = 0
    # Get initial Mask
    for i in range(config.get('InitialMaskFrames')): reader.readFrame()
    reader.updateMask()
    # Start Loop
    while True:
        # Read frame
        reader.readFrame()
        frame = reader.getLatestFrame()
        # Automatic whitebalance
        if config.get('Autowhitebalance') != 0: frame = transformer.applyWhitebalance(frame)
        # Alter colour temperature
        if config.get('ColourTemp') != 0: frame = transformer.applyColourTemp()
        # Gamma correction
        if config.get('Gamma') != 1: frame = transformer.applyGammaCorrection(frame)
        # Alter saturation
        if config.get('Saturation') != 1: frame = transformer.applySaturation(frame, config.get('Saturation'))
        # Get LED colours
        data = reader.getLEDColours(frame, numLEDs=config.get('numLEDs'))
        # Apply motion smoothing
        data = transformer.applyMotionSmoothing(data)
        # Send LED data to arduino
        if config.get('invertLEDOrder') == 1: data = np.flip(data[0], axis=0)
        else: data = data[0]
        ser.sendLEDData(numLEDs=config.get('numLEDs'), data=data)
        # Update mask in slower intervals to save performance
        if frameCount == config.get('MaskUpdateRate'):
            reader.updateMask
            frameCount = 0
        else:
            frameCount += 1
        # Sleep to limit CPU utilisation
        time.sleep(config.get('SleepBetweenRefreshes'))

if __name__ == "__main__":
    main()
