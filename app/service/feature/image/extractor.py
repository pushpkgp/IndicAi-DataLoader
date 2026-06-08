import cv2
import numpy as np
from skimage.exposure import exposure
from skimage.restoration import denoise_wavelet

from app.model.fcn import fcn_model, apply_mask
from app.optimization.sa_bbo import sa_bbo
from app.service.feature.image.features import extract_features
from app.service.feature.preprocessor import logger

# Transform Image
def transform(image_path):
    return cv2.cvtColor(cv2.resize(cv2.imread(image_path), (256, 256)), cv2.COLOR_BGR2GRAY)

# Noise reduction using wavelet thresholding
def denoise(image):
    denoised_image = denoise_wavelet(image) * 255
    # cv2.imwrite('Pictorial Results/3. Denoised Image.jpg', denoised_image)
    return denoised_image.astype('uint8')

# Intensity normalization
def normalize(denoised_image):
    normalized_image = cv2.normalize(denoised_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imwrite('Pictorial Results/4. Normalized Image.jpg', normalized_image)
    return normalized_image

# Histogram equalization to enhance feature visibility
def histogram_equalize(normalized_image):
    equalized_image = exposure.equalize_hist(normalized_image) * 255
    equalized_image = cv2.cvtColor(equalized_image.astype('uint8'), cv2.COLOR_GRAY2BGR)
    # cv2.imwrite('Pictorial Results/5. Equalized Image.jpg', equalized_image)
    return equalized_image

def segment(equalized_image, model):
    # Predict segmentation mask using the FCN model
    logger.info(f"Pushpinder Features: In Segmentation")
    segmentation_mask = model.predict(np.expand_dims(equalized_image, axis=0))

    # Apply segmentation mask to input image
    segmented_image = apply_mask(equalized_image, segmentation_mask[0])
    # cv2.imwrite('Pictorial Results/6. Segmented Image.jpg', segmented_image)
    logger.info(f"Pushpinder Features: Segmentation Complete")
    return segmented_image

def extract_image_features(image_path: str, model):
    logger.info(f"Pushpinder Extracting Image Features for image {image_path}, model: {model}")
    features = extract_features(segment(histogram_equalize(normalize(denoise(transform(image_path)))), model))
    logger.info(f"Pushpinder Features {features} for file {image_path}")
    return features