import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def apply_mask(image, mask, threshold: float = 0.5):
    if mask.ndim == 3 and mask.shape[-1] > 1:
        mask_binary = np.max(mask, axis=-1)
    elif mask.ndim == 3:
        mask_binary = mask[..., 0]
    else:
        mask_binary = mask

    mask_binary = (mask_binary > threshold).astype(np.uint8)

    if image.ndim == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image

    segmented_image = cv2.bitwise_and(image_gray, image_gray, mask=mask_binary)
    return segmented_image

def fcn_model(
    input_shape=(256, 256, 3),
    dilation_rate=(1, 1),
    learning_rate=1e-4,
    num_filters=64,
    kernel_size=3,
    dropout_rate=0.2,
):
    inputs = tf.keras.layers.Input(shape=input_shape)

    f1 = int(num_filters)
    f2 = min(f1 * 2, 256)
    f3 = min(f1 * 4, 512)

    conv1 = tf.keras.layers.Conv2D(
        f1, (kernel_size, kernel_size), activation="relu",
        padding="same", dilation_rate=dilation_rate
    )(inputs)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Dropout(dropout_rate)(conv1)

    conv2 = tf.keras.layers.Conv2D(
        f1, (kernel_size, kernel_size), activation="relu", padding="same"
    )(conv1)
    pool1 = tf.keras.layers.AveragePooling2D((2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(
        f2, (kernel_size, kernel_size), activation="relu", padding="same"
    )(pool1)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)

    conv4 = tf.keras.layers.Conv2D(
        f2, (kernel_size, kernel_size), activation="relu", padding="same"
    )(conv3)
    pool2 = tf.keras.layers.AveragePooling2D((2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(
        f3, (kernel_size, kernel_size), activation="relu", padding="same"
    )(pool2)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)

    conv6 = tf.keras.layers.Conv2D(
        f3, (kernel_size, kernel_size), activation="relu", padding="same"
    )(conv5)

    upsample1 = tf.keras.layers.Conv2DTranspose(
        f2, (kernel_size, kernel_size), strides=(2, 2), padding="same"
    )(conv6)
    upsample1 = tf.keras.layers.Concatenate(axis=3)([conv4, upsample1])

    conv7 = tf.keras.layers.Conv2D(
        f2, (kernel_size, kernel_size), activation="relu", padding="same"
    )(upsample1)
    conv8 = tf.keras.layers.Conv2D(
        f2, (kernel_size, kernel_size), activation="relu", padding="same"
    )(conv7)

    upsample2 = tf.keras.layers.Conv2DTranspose(
        f1, (kernel_size, kernel_size), strides=(2, 2), padding="same"
    )(conv8)
    upsample2 = tf.keras.layers.Concatenate(axis=3)([conv2, upsample2])

    conv9 = tf.keras.layers.Conv2D(
        f1, (kernel_size, kernel_size), activation="relu", padding="same"
    )(upsample2)
    conv10 = tf.keras.layers.Conv2D(
        f1, (kernel_size, kernel_size), activation="relu", padding="same"
    )(conv9)

    # Binary lung mask output.
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(conv10)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model