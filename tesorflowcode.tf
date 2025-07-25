import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# ======================================
# 1. Dataset Preparation and Augmentation
# ======================================
IMG_SIZE = (224, 224)  # Standard size for pretrained models
BATCH_SIZE = 32

# Define dataset paths
dataset_dir = "nail_dataset"
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "validation")
test_dir = os.path.join(dataset_dir, "test")

# Advanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Validation and test generators (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
)

# ======================================
# 2. Advanced Model Architecture
# ======================================
def build_advanced_model(input_shape):
    # Use EfficientNet as base
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze base layers
    base_model.trainable = False
    
    # Create new model on top
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

model = build_advanced_model((IMG_SIZE[0], IMG_SIZE[1], 3))

# ======================================
# 3. Model Compilation with Callbacks
# ======================================
initial_learning_rate = 0.001
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=lr_schedule),
    loss='binary_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model_checkpoint = callbacks.ModelCheckpoint(
    'best_nail_detector.h5',
    save_best_only=True,
    monitor='val_accuracy'
)

tensorboard = callbacks.TensorBoard(log_dir='./logs')

# ======================================
# 4. Model Training
# ======================================
EPOCHS = 50

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint, tensorboard]
)

# ======================================
# 5. Model Evaluation and Visualization
# ======================================
def plot_training_history(history):
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        plt.plot(history.history[metric], label='Training')
        plt.plot(history.history[f'val_{metric}'], label='Validation')
        
        plt.title(f'Model {metric}')
        plt.ylabel(metric)
        plt.xlabel('Epoch')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Load best model
model = tf.keras.models.load_model('best_nail_detector.h5')

# Evaluate on test set
test_results = model.evaluate(test_generator)
print(f"Test Accuracy: {test_results[1]*100:.2f}%")
print(f"Test Precision: {test_results[2]*100:.2f}%")
print(f"Test Recall: {test_results[3]*100:.2f}%")

# ======================================
# 6. Advanced Model Fine-Tuning
# ======================================
# Unfreeze some layers for fine-tuning
model.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Fine-tune for a few epochs
fine_tune_epochs = 10
total_epochs = EPOCHS + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[early_stopping, model_checkpoint, tensorboard]
)

# Save final model
model.save('final_nail_detector.h5')

# ======================================
# 7. Prediction and Visualization
# ======================================
def predict_and_visualize(model, generator, num_samples=5):
    # Get sample batch
    x_batch, y_batch = next(generator)
    
    # Make predictions
    predictions = model.predict(x_batch)
    
    plt.figure(figsize=(15, 10))
    
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(x_batch[i])
        
        pred = "Nail" if predictions[i] > 0.5 else "No Nail"
        true = "Nail" if y_batch[i] > 0.5 else "No Nail"
        
        plt.title(f"Pred: {pred}\nTrue: {true}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
# Visualize predictions
predict_and_visualize(model, test_generator)
