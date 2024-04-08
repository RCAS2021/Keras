import numpy as np
import os
import tensorflow

# Setting the backed for Keras to TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

# Loading the data and splitting it between train and test sets
(train_x, train_y) , (test_x, test_y) = keras.datasets.mnist.load_data()

# Scaling images to the [0, 1] range
train_x = train_x.astype("float32") / 255
test_x = test_x.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
train_x = np.expand_dims(train_x, -1)
test_x = np.expand_dims(test_x, -1)
print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")
print(f"{train_x.shape[0]} trains samples")
print(f"{test_x.shape[0]} test samples")

num_classes = 10
input_shape = (28, 28, 1)

# Define the neural network model architecture using Sequential API
model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

# Print model summary
model.summary()

# Compile the model with specified loss function, optimizer and metrics
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc"),],
)

batch_size = 128
epochs = 5

# Define callbacks for model training
callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="models/model_at_epoch_{epoch}.keras"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]

# Train the model using training data
model.fit(
    train_x,
    train_y,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
    callbacks=callbacks,
)

# Evaluate the trained model on test data
score = model.evaluate(test_x, test_y, verbose=1)

# Save the trained model
model.save("models/final_model.keras")

# Load the saved model
model = keras.saving.load_model("models/final_model.keras")

# Make predictions using the loaded model
prediction = model.predict(test_x)
