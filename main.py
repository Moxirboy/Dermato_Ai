import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from flask import Flask, request, jsonify, render_template
from flask import request as rq
import requests 
import json
app = Flask(__name__)

# Loading the model
model = load_model('model.h5', compile=False)

# Compiling the model
optimizer = Adam(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Seven classes
class_names = {
    0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion'
}

# Process image
def preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize image to [0, 1]
    return img_array

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in rq.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = rq.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    target_size = (224, 224)
    img_array = preprocess_image(file_path, target_size)

    # Predictions
    predictions = model.predict(img_array)
    threshold = 0.5

    # Predicted class
    predicted_class_index = np.argmax(predictions)
    predicted_class_confidence = predictions[0][predicted_class_index]

    # Statement to determine the illness
    if predicted_class_confidence > threshold:
        predicted_class_name = class_names[predicted_class_index]
        result = {
            'Predicted Class': predicted_class_name,
            'Confidence': f"{predicted_class_confidence:.2f}"
        }
    else:
        result = {'message': 'No illness detected with sufficient confidence.'}

    # Probability for each class
    probabilities = {class_names[idx]: f"{predictions[0][idx]:.2f}" for idx in class_names}
    result['Probabilities'] = probabilities
    data={"request":json.dumps(result)+"if you get this type of prompt, this is response from our skin scanner ai, so you have to give to user some info about result, confidence is the prediction percentage, predicted class is the diagnose, you say your diagnose is \"predicted class\", this prediction has accuracy \"confidence\", than you give short info about the illness and recommend specific doctor to visit(like dermatologist and etc), in the end you include privacy and policy message, like the ai is not 100 percent correct and it cannot detect other objects and some times can give false info and that ai doesn't take any responsibilities."
     } 
    external_url = 'https://appai-production.up.railway.app/generate_response'  # Replace with your URL
    try:
        response = requests.post(external_url, json=data)
        response.raise_for_status()
        external_result = response.json()
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error communicating with external service: {str(e)}'}), 500

    return  external_result

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5005)
