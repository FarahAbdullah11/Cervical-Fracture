from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Preprocessing + Augmentation
train_datagen = ImageDataGenerator(

    # Normalize pixel values
    rescale=1./255,

    # Split dataset into training and validation
    validation_split=0.2,

    # Data Augmentation
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Training Data
train_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation Data
val_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Check classes
print(train_data.class_indices)