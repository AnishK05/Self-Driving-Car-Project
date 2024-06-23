import socketio  # Importing the Socket.IO library for real-time communication
import eventlet  # Importing Eventlet for asynchronous networking support
import numpy as np  # Importing NumPy for numerical operations
from flask import Flask  # Importing Flask to create the web server
from keras.models import load_model  # Importing Keras to load the trained model
import base64  # Importing base64 to decode image data received from the client
from io import BytesIO  # Importing BytesIO to handle image data as bytes
from PIL import Image  # Importing PIL (Python Imaging Library) for image processing
import cv2  # Importing OpenCV for additional image processing

# Initialize a new Socket.IO server
sio = socketio.Server()

# Initialize a new Flask web application
app = Flask(__name__)  # The '__name__' variable indicates the current module

# Define a speed limit for the car
speed_limit = 10

# Function to preprocess the image received from the simulator -- same as in CNN model
def image_preprocess(image):
    image = image[60:135,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255
    return image

# Event handler for telemetry data from the car simulator
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])  # Get the current speed of the vehicle
    image = Image.open(BytesIO(base64.b64decode(data['image'])))  # Decode the base64 image data
    image = np.asarray(image)  # Convert the image to a NumPy array
    image = image_preprocess(image)  # Preprocess the image
    image = np.array([image])  # Add a batch dimension to the image
    steering_angle = float(model.predict(image))  # Predict the steering angle using the pre-trained model
    throttle = 1.0 - speed / speed_limit  # Calculate the throttle to maintain the speed limit
    print('{} {} {}'.format(steering_angle, throttle, speed))  # Print the steering angle, throttle, and speed
    send_control(steering_angle, throttle)  # Send the control commands back to the simulator

# Event handler for a new connection from the car simulator
@sio.on('connect')
def connect(sid, environ):
    print('Connected')  # Print a message when a new connection is established
    send_control(0, 0)  # Send initial control commands to the simulator

# Function to send control commands to the car simulator
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),  # Convert the steering angle to a string and send it
        'throttle': throttle.__str__()  # Convert the throttle to a string and send it
    })

# Main block to start the server
if __name__ == '__main__':
    model = load_model('models/model.h5')  # Load the pre-trained Keras model
    app = socketio.Middleware(sio, app)  # Wrap the Flask app with Socket.IO middleware
    # Start the Eventlet WSGI server on port 4567; Port was found using netstart -ano | findstr <PID> on admin terminal
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app) # Eventlet can handle multiple concurrent connections efficiently 