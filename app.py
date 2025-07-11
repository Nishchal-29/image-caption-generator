import os
import numpy as np
import pickle
from flask import Flask, render_template, redirect, request
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

caption_model = load_model('models/caption_model2.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    
img_model = DenseNet201()
model = Model(inputs=img_model.input, outputs=img_model.layers[-2].output)

def extract_features(filename):
    img = load_img(filename, target_size=(224, 224))
    img = img_to_array(img)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    feature = model.predict(img)
    return feature

MaxLen = 35

def generate_caption(photo):
    in_text = 'startseq'
    for i in range(MaxLen):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen = MaxLen)
        yhat = caption_model.predict([photo, seq], verbose=0)
        yhat = np.argmax(yhat)
        word = None
        for w, idx in tokenizer.word_index.items():
            if idx == yhat:
                word = w
                break
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['image']
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    photo = extract_features(filename=filename)
    caption = generate_caption(photo=photo)
    
    return render_template('predict.html', filename=file.filename, caption=caption)

if __name__ == '__main__':
    app.run(debug=True)