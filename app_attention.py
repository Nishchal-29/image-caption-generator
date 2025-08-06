import os
import numpy as np
import pickle
from flask import Flask, render_template, redirect, request
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from custom_layers import BahdanauAttention, EncoderCNN, DecoderRNN, ImageCaptioningModel, Vocabulary

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

custom_objects = {
    "EncoderCNN": EncoderCNN,
    "DecoderRNN": DecoderRNN,
    "BahdanauAttention": BahdanauAttention,
    "ImageCaptioningModel": ImageCaptioningModel,
}

model = tf.keras.models.load_model("models/final_caption_model.keras", compile=False, custom_objects=custom_objects)
with open('models/vocab_attention.pkl', 'rb') as f:
    vocab_attention = pickle.load(f)
model.vocab = vocab_attention
    
def preprocess_image_tensor(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    return tf.convert_to_tensor(arr)

max_length = 19
beam_width = 3

def generate_caption(img_tensor):
    start = vocab_attention.word2idx["<start>"]
    end = vocab_attention.word2idx["<end>"]
    encoder = model.encoder
    decoder = model.decoder
    features = encoder(img_tensor)
    hidden, cell = decoder.reset_state(batch_size=1)
    beams = [([start], hidden, cell, 0.0)]
    for _ in range(max_length - 1):
        candidates = []
        for seq, h, c, score in beams:
            if seq[-1] == end:
                candidates.append((seq, h, c, score))
                continue
            dec_input = tf.expand_dims([seq[-1]], 1)
            predictions, h, c, _ = decoder(dec_input, features, h, c, training = False)
            log_probs = tf.nn.log_softmax(predictions, axis = -1)[0].numpy()
            top_idxs = np.argsort(log_probs)[-beam_width:]
            for idx in top_idxs:
                new_seq = seq + [int(idx)]
                new_score = score + log_probs[idx]
                candidates.append((new_seq, h, c, new_score))
        beams = sorted(candidates, key = lambda x : x[3] / len(x[0]), reverse = True)[:beam_width]
        if all(b[0][-1] == end for b in beams):
            break
    best_seq = beams[0][0]
    tokens = [vocab_attention.idx2word.get(i, "<unk>") for i in best_seq if i not in {start, end}]
    return " ".join(tokens)

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
    img_tensor = preprocess_image_tensor(filename)
    caption = generate_caption(img_tensor)
    return render_template('predict.html', filename=file.filename, caption=caption)

if __name__ == '__main__':
    app.run(debug=True)