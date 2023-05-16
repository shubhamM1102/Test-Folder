import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data paths
# train_path = os.getcwd() + "\\gesture\\train"
# test_path = os.getcwd() + "\\gesture\\test"

train_data_path = os.getcwd() + "\\gesture\\train"
val_data_path = os.getcwd() + "\\gesture\\validate"
test_data_path = os.getcwd() + "\\gesture\\test"

# Define image dimensions and batch size
img_width, img_height = 200, 200
batch_size = 32

# Define data generators with data augmentation for training and validation data
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary')

# Set up data generator for validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    val_data_path,
    target_size=(200,200),
    batch_size=32,
    class_mode='binary')

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

# Train the model using the generator
history = model.fit(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1)

# Evaluate model on test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print('Test accuracy:', test_acc)
model.save('best_model_dataflair3.h5')