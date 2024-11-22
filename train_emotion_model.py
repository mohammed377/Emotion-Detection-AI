import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json

# Directories
train_dir = r"C:\Users\mohammed ismail\OneDrive\Desktop\AI\TASK2\emotions_dataset\train"
test_dir = r"C:\Users\mohammed ismail\OneDrive\Desktop\AI\TASK2\emotions_dataset\test"
image_size = (48, 48)
batch_size = 32

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

# Model Architecture
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Flatten(),
    Dropout(0.5),
    
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile the Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
model_checkpoint = ModelCheckpoint('best_emotion_model.h5', save_best_only=True)

# Train the Model
model.fit(train_generator, 
          steps_per_epoch=train_generator.samples // batch_size,
          validation_data=test_generator, 
          validation_steps=test_generator.samples // batch_size, 
          epochs=30, 
          callbacks=[early_stopping, reduce_lr, model_checkpoint])

# Save the Model and Class Labels
model.save('emotion_model.h5')
class_labels = {v: k for k, v in train_generator.class_indices.items()}
with open('class_labels.json', 'w') as json_file:
    json.dump(class_labels, json_file)

print("Model and class labels saved successfully!")
