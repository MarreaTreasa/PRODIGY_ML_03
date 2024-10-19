from flask import Flask, request, render_template
import numpy as np
import cv2
import os
import pickle

app = Flask(__name__)

with open('info.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def prepare_image(image):
    img = cv2.imread(image, 0)
    img = cv2.resize(img, (50, 50))
    img_flat = img.flatten() / 255.0  
    return np.array([img_flat])  

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        image_data = prepare_image(filepath)
        prediction = model.predict(image_data)
        os.remove(filepath)  
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
