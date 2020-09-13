from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os

import numpy as np
import tensorflow.keras
from PIL import Image, ImageOps

import random

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
PREDICTION_THRESHOLD = .6

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
  prediction_text_one = truncate(prediction.item(0)*100, 5)
  prediction_text_two = truncate(prediction.item(1)*100, 5)
  prediction_text_three = truncate(prediction.item(2)*100, 5)

  # 1 is rock
  # 2 is scissor
  # 3 is paper
  num = random.randint(1,3)

  if prediction.item(0) > PREDICTION_THRESHOLD:
    if num == 1:
      return "this is {}% a {}".format(prediction_text_one, "rock. mine is rock too, we're draw.")
    elif num == 2:
      return "this is {}% a {}".format(prediction_text_one, "rock. mine is scissor, you win.")
    elif num == 3:
      return "this is {}% a {}".format(prediction_text_one, "rock. mine is paper, i win.")
  else:
    return "this is {}% a {}".format(prediction_text_one, "shit wtf u just uploaded")

  if prediction.item(1) > PREDICTION_THRESHOLD:
    if num == 1:
      return "this is {}% a {}".format(prediction_text_one, "scissor. mine is rock, i win.")
    elif num == 2:
      return "this is {}% a {}".format(prediction_text_one, "scissor. mine is scissor too, we're draw.")
    elif num == 3:
      return "this is {}% a {}".format(prediction_text_one, "scissor. mine is paper, you win.")
  else:
    return "this is {}% a {}".format(prediction_text_one, "shit wtf u just uploaded")


  if prediction.item(2) > PREDICTION_THRESHOLD:
    if num == 1:
      return "this is {}% a {}".format(prediction_text_one, "paper. mine is rock, you win.")
    elif num == 2:
      return "this is {}% a {}".format(prediction_text_one, "paper. mine is scissor, i win.")
    elif num == 3:
      return "this is {}% a {}".format(prediction_text_one, "paper. mine is paper too, we're draw.")
  else:
    return "this is {}% a {}".format(prediction_text_one, "shit wtf u just uploaded")


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)

    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)

    i, p, d = s.partition('.')

    return '.'.join([i, (d+'0'*n)[:n]])




if __name__ == "__main__":
    app.run(debug=True)