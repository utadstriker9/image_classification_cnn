from flask import Flask, jsonify, request
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image
import utils

CONFIG_DATA = utils.config_load()

app = Flask(__name__)

# Load the saved model
saved_model_path = CONFIG_DATA['model_path']  # Path to the saved model
model = load_model(saved_model_path)

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']
    image = Image.open(image_file)
    
    # Preprocess the image
    image = image.resize(resize_width=CONFIG_DATA['resize_width'], resize_height=CONFIG_DATA['resize_height'])
    image = img_to_array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make a prediction
    prediction = model.predict(image)
    
    # Get the predicted label
    predicted_label = np.argmax(prediction, axis=1)[0]
    
    # Return the predicted label as a JSON response
    labels = ['Banana', 'Apple', 'Grape']
    response = {'predicted_label': labels[predicted_label]}
    return jsonify(response)

if __name__ == '__main__':
    # Set the IP address and port number
    ip_address = '0.0.0.0'  # Set to '0.0.0.0' to bind to all available network interfaces
    port = 5000  # Set the desired port number
    
    # Start the Flask development server
    app.run(host=ip_address, port=port)
