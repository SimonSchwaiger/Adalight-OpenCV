#!/env/bin/python

import paho.mqtt.client as mqtt
# https://pypi.org/project/paho-mqtt/
# https://community.home-assistant.io/t/create-a-button-to-publish-to-mqtt/239077

import json
import time
import subprocess
import sys

# Load configuration
with open('./config.json') as json_file:
    config = json.load(json_file)

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("livingRoom/adalight")

# The callback for when a PUBLISH message is received from the server.
class onMessageCallback(object):
    def __init__(self, data=False):
        self.p = None
    #
    def __call__(self, client, userdata, msg):
        if msg.payload == b'on':
            # If payload is 'on', then start a subprocess running the ambilight app
            # https://stackoverflow.com/questions/546017/how-do-i-run-another-script-in-python-without-waiting-for-it-to-finish
            if self.p == None:
                self.p = subprocess.Popen(
                    [sys.executable, 'application.py'], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT
                )
        else:
            # Otherwise, stop the process
            if self.p != None: 
                self.p.terminate()
                self.p = None
        # Print Payload for debugging
        print(msg.topic+" "+str(msg.payload))

on_message = onMessageCallback()

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Set username and password and connect
client.username_pw_set(
    username = config.get('MQTTCredentials')[0],
    password = config.get('MQTTCredentials')[1]
)

client.connect(
    config.get('MQTTIP'),
    1883,
    60
)

# Loop forever and wait for messages
client.loop_forever()


#while True:
#    if on_message.run:
#        # Run Program
#        pass
#    else:
#        # Stop Program
#        pass
#    time.sleep(0.5)
#    print("Update loop. run: {}".format(on_message.run))
#    client.loop()



# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
#client.loop_forever()