import tensorflow as tf
import numpy as np

# Sample dataset
X_train = np.random.rand(100, 64, 64, 3).astype("float32")
y_train = np.random.randint(0, 2, 100)

# ✅ Custom Model Class
class FruitRipenessModel(tf.keras.Model):
    def __init__(self):
        super(FruitRipenessModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, 3, activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.out(x)

    # ✅ Exportable serving function
    @tf.function(input_signature=[tf.TensorSpec([None, 64, 64, 3], tf.float32)])
    def serving(self, inputs):
        return {"predictions": self.call(inputs)}

# Instantiate and compile
model = FruitRipenessModel()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=2)

# ✅ Export with correct signature
tf.saved_model.save(model, "model", signatures={"serving_default": model.serving})
