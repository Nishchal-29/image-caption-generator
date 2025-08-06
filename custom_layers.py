import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.idx = 4

    def tokenizer(self, text):
        return [token.text.lower() for token in nlp(text)]

    def build_vocab(self, sentences):
        frequencies = Counter()
        for sentence in sentences:
            frequencies.update(self.tokenizer(sentence))
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def numericalize(self, text):
        tokens = self.tokenizer(text)
        return [self.word2idx.get(t, self.word2idx["<unk>"]) for t in tokens]

@register_keras_serializable(package="custom", name="EncoderCNN")
class EncoderCNN(Model):
    def __init__(self, embedding_dim, train_backbone=False, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.train_backbone = train_backbone
        self.base = MobileNetV2(include_top=False, weights="imagenet", pooling="avg")
        self.base.trainable = train_backbone
        self.fc = layers.Dense(embedding_dim)

    def call(self, x):
        x = self.base(x)
        return self.fc(x)

    def get_config(self):
        config = super().get_config()
        config.update({"embedding_dim": self.embedding_dim, "train_backbone": self.train_backbone})
        return config

    @classmethod
    def from_config(cls, config):
        embedding_dim = config.pop("embedding_dim")
        train_backbone = config.pop("train_backbone")
        return cls(embedding_dim=embedding_dim, train_backbone=train_backbone, **config)
     
@register_keras_serializable(package="custom", name="BahdanauAttention")
class BahdanauAttention(Model):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features, hidden):
        hidden_time = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_time))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config):
        units = config.pop("units")
        return cls(units=units, **config)
    
@register_keras_serializable(package="custom", name="DecoderRNN")
class DecoderRNN(Model):
    def __init__(self, vocab_size, embedding_dim, units, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.units = units
        self.dropout_rate = dropout_rate
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.attention = BahdanauAttention(units)
        self.dropout_embed = layers.Dropout(dropout_rate)
        self.lstm = layers.LSTM(units, return_sequences=True, return_state=True)
        self.dropout_lstm = layers.Dropout(dropout_rate)
        self.fc = layers.Dense(vocab_size)

    def call(self, x, features, hidden, cell, training=False):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = self.dropout_embed(x, training=training)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state_h, state_c = self.lstm(x, initial_state=[hidden, cell])
        output = self.dropout_lstm(output, training=training)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state_h, state_c, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "units": self.units,
            "dropout_rate": self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        vocab_size = config.pop("vocab_size")
        embedding_dim = config.pop("embedding_dim")
        units = config.pop("units")
        dropout_rate = config.pop("dropout_rate")
        return cls(vocab_size=vocab_size, embedding_dim=embedding_dim, units=units, dropout_rate=dropout_rate, **config)
    
@register_keras_serializable(package="custom", name="ImageCaptioningModel")
class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, encoder, decoder, vocab, max_length, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab 
        self.max_length = max_length

    def call(self, inputs, training=False):
        img_tensor, input_seq = inputs
        batch_size = tf.shape(input_seq)[0]
        seq_len = tf.shape(input_seq)[1]
        features = self.encoder(img_tensor)
        hidden, cell = self.decoder.reset_state(batch_size)
        dec_input = tf.expand_dims(input_seq[:, 0], 1)
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        i = tf.constant(0)

        def cond(i, dec_input, hidden, cell, outputs_ta):
            return tf.less(i, seq_len)

        def body(i, dec_input, hidden, cell, outputs_ta):
            predictions, hidden, cell, _ = self.decoder(dec_input, features, hidden, cell, training=training)
            outputs_ta = outputs_ta.write(i, predictions)
            next_i = i + 1
            dec_input_next = tf.cond(
                tf.less(next_i, seq_len),
                lambda: tf.expand_dims(input_seq[:, next_i], 1),
                lambda: dec_input
            )
            return next_i, dec_input_next, hidden, cell, outputs_ta

        _, _, _, _, outputs_ta = tf.while_loop(
            cond, body,
            loop_vars=[i, dec_input, hidden, cell, outputs_ta],
            shape_invariants=[
                i.get_shape(),
                tf.TensorShape([None, 1]),
                tf.TensorShape([None, self.decoder.units]),
                tf.TensorShape([None, self.decoder.units]),
                tf.TensorShape(None)
            ]
        )
        outputs = outputs_ta.stack()
        outputs = tf.transpose(outputs, [1, 0, 2])
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_length": self.max_length,
            "encoder": {
                "class_name": self.encoder.__class__.__name__,
                "config": self.encoder.get_config()
            },
            "decoder": {
                "class_name": self.decoder.__class__.__name__,
                "config": self.decoder.get_config()
            }
        })
        return config

    @classmethod
    def from_config(cls, config):
        max_length = config.pop("max_length")
        encoder_dict = config.pop("encoder")
        decoder_dict = config.pop("decoder")
        encoder = tf.keras.utils.deserialize_keras_object({"class_name": encoder_dict["class_name"], "config": encoder_dict["config"]})
        decoder = tf.keras.utils.deserialize_keras_object({"class_name": decoder_dict["class_name"], "config": decoder_dict["config"]})
        dummy_vocab = None
        return cls(encoder=encoder, decoder=decoder, vocab=dummy_vocab, max_length=max_length, **config)