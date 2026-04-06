import tensorflow as tf
from tensorflow.keras import layers, models


# -------------------------------
# Attention Layer (Fixed Properly)
# -------------------------------

class AttentionLayer(layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(
            shape=(1,),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs):
        score = tf.matmul(inputs, self.W) + self.b
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * inputs
        return tf.reduce_sum(context_vector, axis=1)

# -------------------------------
# Residual Block (Lightweight)
# -------------------------------
def res_block(x, filters):
    shortcut = x

    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), padding='same')(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x


# -------------------------------
# FINAL MODEL
# -------------------------------
def build_next_model(input_shape, num_classes):

    inp = layers.Input(shape=input_shape)

    # ---------------- CNN (Reduced depth)
    x = res_block(inp, 32)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = res_block(x, 64)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = res_block(x, 128)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.4)(x)

    # ---------------- reshape for LSTM
    x = layers.Permute((2,1,3))(x)
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # ---------------- BiLSTM (reduced)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True)
    )(x)

    # ---------------- Attention
    x = AttentionLayer()(x)

    # ---------------- Dense
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model