pip install tensorflow numpy matplotlib scikit-learn

# Importing Essential Libraries
import os
import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception  # You can switch to MobileNetV2 for a lighter model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# (Optional) If you plan to use mixed precision training
# from tensorflow.keras import mixed_precision


# Configuration Parameters
BASE_DIR = r"C:\Users\tamim\deepfake detection\dataset"  # Update this path to where your dataset is located
IMAGE_SIZE = (224, 224)  # Reduced from (299, 299) for faster training
BATCH_SIZE = 16  # Reduced from 32 to manage memory and speed up training
INITIAL_EPOCHS = 10  # Number of epochs for initial training
FINE_TUNE_EPOCHS = 10  # Number of epochs for fine-tuning
MODEL_SAVE_PATH = 'best_deepfake_detector.h5'  # Path to save the best model
FINAL_MODEL_SAVE_PATH = 'final_deepfake_detector.h5'  # Path to save the final model

def verify_directory_structure(base_dir=BASE_DIR, subsets=['train', 'validation'], classes=['real', 'fake']):
    """
    Verifies that the dataset directory structure exists.

    Parameters:
    - base_dir (str): Base directory of the dataset.
    - subsets (list): List of dataset subsets (e.g., train, validation).
    - classes (list): List of classes (e.g., real, fake).

    Raises:
    - FileNotFoundError: If any required directory is missing.
    """
    missing_dirs = []
    for subset in subsets:
        for cls in classes:
            dir_path = os.path.join(base_dir, subset, cls)
            if not os.path.isdir(dir_path):
                missing_dirs.append(dir_path)
            else:
                print(f"Directory exists: {dir_path}")
    if missing_dirs:
        raise FileNotFoundError(f"The following directories are missing: {missing_dirs}")

# Verify directory structure
verify_directory_structure()



def create_data_generators(base_dir=BASE_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    """
    Creates training and validation data generators.

    Parameters:
    - base_dir (str): Base directory of the dataset.
    - image_size (tuple): Target size of the images.
    - batch_size (int): Number of samples per batch.

    Returns:
    - train_generator, validation_generator: Keras data generators.
    """
    train_dir = os.path.join(base_dir, 'train')
    
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"The directory {train_dir} does not exist. Please check the path.")
    
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        fill_mode="nearest",
        validation_split=0.2  # 80-20 split for training and validation
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

# Create data generators
train_generator, validation_generator = create_data_generators()


def build_model(image_size=IMAGE_SIZE):
    """
    Builds and compiles the deepfake detection model using Xception.

    Parameters:
    - image_size (tuple): Input image size.

    Returns:
    - model: Compiled Keras model.
    """
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
    base_model.trainable = False  # Freeze the base model initially
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Regularization
    x = Dense(512, activation='relu')(x)  # Reduced units for faster training
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)  # Binary classification
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    return model

# Build the model
model = build_model()
model.summary()



def train_model(model, base_model, train_gen, val_gen, save_path=MODEL_SAVE_PATH, initial_epochs=INITIAL_EPOCHS, fine_tune_epochs=FINE_TUNE_EPOCHS):
    """
    Trains the model with initial training and fine-tuning phases.

    Parameters:
    - model: Keras model to train.
    - base_model: The base model within the Keras model.
    - train_gen: Training data generator.
    - val_gen: Validation data generator.
    - save_path (str): Path to save the best model.
    - initial_epochs (int): Number of epochs for initial training.
    - fine_tune_epochs (int): Number of epochs for fine-tuning.

    Returns:
    - history: Training history for initial training.
    - history_fine: Training history for fine-tuning.
    """
    # Define callbacks
    checkpoint = ModelCheckpoint(save_path,
                                 monitor='val_auc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    
    early_stop = EarlyStopping(monitor='val_auc',
                               patience=5,  # Reduced patience for quicker stopping
                               verbose=1,
                               restore_best_weights=True)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_auc',
                                  factor=0.2,
                                  patience=3,
                                  verbose=1,
                                  min_lr=1e-6)
    
    callbacks_list = [checkpoint, early_stop, reduce_lr]
    
    # Calculate steps per epoch
    steps_per_epoch = math.ceil(train_gen.samples / train_gen.batch_size)
    validation_steps = math.ceil(val_gen.samples / val_gen.batch_size)
    
    # Initial training: Train the top layers
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=initial_epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks_list
    )
    
    # Fine-Tuning
    base_model.trainable = True  # Unfreeze the base model
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100  # Adjust based on model architecture
    
    # Freeze all layers before the fine_tune_at layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompile the model with a lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    
    # Continue training (Fine-Tuning)
    total_epochs = initial_epochs + fine_tune_epochs  ##CHANGE NO OF EPOCHS FROM HERE!!

    
    history_fine = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks_list
    )
    
    return history, history_fine

# Train the model
history, history_fine = train_model(model, base_model, train_generator, validation_generator)


def plot_training_history(history, history_fine):
    """
    Plots training and validation accuracy and AUC.

    Parameters:
    - history: Training history from initial training.
    - history_fine: Training history from fine-tuning.
    """
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    auc = history.history['auc'] + history_fine.history['auc']
    val_auc = history.history['val_auc'] + history_fine.history['val_auc']
    
    total_epochs = len(acc)
    epochs_range = range(total_epochs)
    
    plt.figure(figsize=(12, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    # AUC Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, auc, label='Training AUC')
    plt.plot(epochs_range, val_auc, label='Validation AUC')
    plt.legend(loc='lower right')
    plt.title('Training and Validation AUC')
    
    plt.show()

# Plot training history
plot_training_history(history, history_fine)



# Save the final model
model.save(FINAL_MODEL_SAVE_PATH)
print(f"Final model saved successfully at {FINAL_MODEL_SAVE_PATH}.")


def load_trained_model(save_path=FINAL_MODEL_SAVE_PATH):
    """
    Loads a trained Keras model from the specified path.

    Parameters:
    - save_path (str): Path to the saved model file.

    Returns:
    - loaded_model: The loaded Keras model.
    """
    if not os.path.isfile(save_path):
        raise FileNotFoundError(f"The model file {save_path} does not exist.")
    loaded_model = tf.keras.models.load_model(save_path)
    print(f"Model loaded successfully from {save_path}.")
    return loaded_model

# Example usage:
# loaded_model = load_trained_model()


def predict_image_interactive(model, image_size=IMAGE_SIZE):
    """
    Prompts the user to input an image path and predicts whether it's Real or Fake.
    
    Parameters:
    - model: Trained Keras model.
    - image_size (tuple): Target size of the image.
    
    Returns:
    - label (str): 'Real' or 'Fake'.
    - confidence (float): Confidence percentage of the prediction.
    """
    # Prompt user for image path
    img_path = input("Please enter the full path to the image you want to classify: ")
    
    if not os.path.isfile(img_path):
        print(f"The image file {img_path} does not exist.")
        return None, None
    
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    img_array /= 255.0  # Rescale
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    label = 'Fake' if prediction < 0.5 else 'Real'
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    
    # Display the image with prediction
    plt.imshow(img)
    plt.title(f"Prediction: {label} ({confidence*100:.2f}% confidence)")
    plt.axis('off')
    plt.show()
    
    print(f"Prediction: {label} ({confidence*100:.2f}% confidence)")
    return label, confidence


# Load the trained model
loaded_model = load_trained_model(FINAL_MODEL_SAVE_PATH)

# Use the interactive prediction function
predict_image_interactive(loaded_model)
