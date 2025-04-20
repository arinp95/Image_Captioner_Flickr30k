from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
import uuid

# App setup
app = Flask(__name__)
upload_dir = 'static/uploads'
app.config['UPLOAD_FOLDER'] = upload_dir
os.makedirs(upload_dir, exist_ok=True)

# Load trained model
model = load_model('model.keras')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load config
with open('config.pkl', 'rb') as f:
    config = pickle.load(f)
max_length = config['max_length']

# Reverse word index
index_word = {v: k for k, v in tokenizer.word_index.items()}

# Feature extractor
densenet = DenseNet201(include_top=False, pooling='avg', input_shape=(224, 224, 3))

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = densenet.predict(img_array, verbose=0)
    return features

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    img = request.files['image']
    if img.filename == '':
        return redirect(url_for('index'))

    if img:
        filename = secure_filename(str(uuid.uuid4()) + '_' + img.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(filepath)

        # Process image and predict
        photo = extract_features(filepath)
        caption = generate_caption(model, tokenizer, photo, max_length)

        return render_template('index.html', caption=caption, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True)
