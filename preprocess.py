import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ukuran gambar & batch size
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

# Load dataset dengan augmentasi
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    'dataset/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Cek label
label_map = train_data.class_indices
print(label_map)

