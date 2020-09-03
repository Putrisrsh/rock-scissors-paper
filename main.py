from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os

import numpy as np
import tensorflow.keras
from PIL import Image, ImageOps

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
PREDICTION_THRESHOLD = .9

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
  return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
      flash('No file part')
      return redirect(request.url)

    file = request.files['file']
    
    if file and allowed_file(file.filename): 
      filename = secure_filename(file.filename) 
      filepath = os.path.join(UPLOAD_FOLDER, filename)
      file.save(filepath)
      prediction = process_file(filepath)
      return render_template('index.html', filename=filename, prediction=prediction)
    else:
      return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

def process_file(filepath):
  model = tensorflow.keras.models.load_model('keras_model.h5')

  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

  image = Image.open(filepath)

  size = (224, 224)
  image = ImageOps.fit(image, size, Image.ANTIALIAS)

  image_array = np.asarray(image)

  image.show()

  normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

  data[0] = normalized_image_array

  prediction = model.predict(data)
  return "this is the output ", prediction


if __name__ == "__main__":
    app.run(debug=True)