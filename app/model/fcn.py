import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.optimizers import Adam

# Function to apply segmentation mask to input image
def apply_mask(image, mask):
    # Convert mask to binary (if necessary)
    mask_binary = np.argmax(mask, axis=-1) if mask.shape[-1] > 1 else (mask > 0.5).astype(np.uint8)
    # Apply mask to input image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask_binary = mask_binary.astype('uint8')
    segmented_image = cv2.bitwise_and(image, image, mask=mask_binary)
    return segmented_image

# Define the FCN model
def fcn_model(input_shape, dilation_rate, learning_rate):
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Convolutional layers  # dilation_rate added
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=(dilation_rate, dilation_rate))(inputs)
    conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.AveragePooling2D((2, 2))(conv2) # instead of max pooling

    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    pool2 = tf.keras.layers.AveragePooling2D((2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    # Transposed convolutional layers for upsampling
    upsample1 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv6)
    upsample1 = tf.keras.layers.concatenate([conv4, upsample1], axis=3)

    conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(upsample1)
    conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    upsample2 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(conv8)
    upsample2 = tf.keras.layers.concatenate([conv2, upsample2], axis=3)

    conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(upsample2)
    conv10 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    # Output layer
    outputs = tf.keras.layers.Conv2D(3, (1, 1), activation='softmax')(conv10)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Compile the model -  tune the learning rate
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Print model summary
    model.summary()

    return model
