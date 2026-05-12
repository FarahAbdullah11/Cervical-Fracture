import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing import train_data, val_data

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Load pretrained ResNet50
base_model = ResNet50(

    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze pretrained layers
base_model.trainable = False

# Build model
model = Sequential([

    base_model,

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(

    optimizer=Adam(learning_rate=0.0001),

    loss='binary_crossentropy',

    metrics=['accuracy']
)

# Model summary
model.summary()

history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=10
)
loss, accuracy = model.evaluate(val_data)

print("Validation Accuracy:", accuracy)