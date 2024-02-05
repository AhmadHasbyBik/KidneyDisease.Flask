from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

app = Flask(__name__)

class_names = ["Cyst", "Normal", "Stone", "Tumor"]

# Define the image size
IMAGE_SIZE = 128

# Load the CNN model
cnn_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu')
])

try:
    cnn_model.load_weights('cnn_model_weights_1.h5')
except Exception as e:
    print(f"Error loading CNN model weights: {str(e)}")

# Define the ELM class
class ELM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, hidden_size)
        self.bias = np.random.randn(hidden_size)
        self.beta = None

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def train(self, X, Y):
        H = self._sigmoid(np.dot(X, self.weights) + self.bias)
        self.beta = np.dot(np.linalg.pinv(H), Y)

    def predict(self, X):
        H = self._sigmoid(np.dot(X, self.weights) + self.bias)
        Y_pred = np.dot(H, self.beta)
        return Y_pred

# Load the ELM model
try:
    with open('elm_model_1.pkl', 'rb') as file:
        elm_model = pickle.load(file)
except Exception as e:
    print(f"Error loading ELM model: {str(e)}")

def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

@app.route('/', methods=['GET'])
def home():
    return "KIDNEY DISEASE DETECTION CNN-ELM"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image was sent with the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'})

        # Get the image file from the request
        image_file = request.files['image']

        # Check if the file is empty
        if image_file.filename == '':
            return jsonify({'error': 'No image provided'})

        # Preprocess the image
        img_array = preprocess_image(image_file)

        if img_array is None:
            return jsonify({'error': 'Error preprocessing image'})

        # Get features from the CNN model
        features_cnn = cnn_model.predict(img_array)

        # Get prediction from the ELM model
        prediction_elm = elm_model.predict(features_cnn.reshape(1, -1))
        predicted_label = np.argmax(prediction_elm)

        # Map predicted label to class name
        predicted_class = class_names[predicted_label]

        # Calculate and display accuracy
        accuracy = np.max(prediction_elm) * 100

        # Construct the response
        response = {
            'prediction': predicted_class,
            'accuracy': accuracy
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
