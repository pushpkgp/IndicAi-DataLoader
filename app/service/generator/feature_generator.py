import cv2
import numpy as np
from skimage.exposure import exposure
from skimage.restoration import denoise_wavelet

from app.model.fcn import fcn_model, apply_mask
from app.optimization.sa_bbo import sa_bbo
from app.service.generator.feature_extractor import extract_features

# Transform Image
def transform(image):
    return cv2.cvtColor(cv2.resize(cv2.imread(image), (256, 256)), cv2.COLOR_BGR2GRAY)

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

def segment(equalized_image):
    # Predict segmentation mask using the FCN model
    segmentation_mask = segmentation_model().predict(np.expand_dims(equalized_image, axis=0))

    # Apply segmentation mask to input image
    segmented_image = apply_mask(equalized_image, segmentation_mask[0])
    # cv2.imwrite('Pictorial Results/6. Segmented Image.jpg', segmented_image)
    return segmented_image

def segmentation_model():
    input_shape = (256, 256, 3)

    # Dilation and Learning rates for parameter tuning in FCN
    lb = [2, 0.0001]
    ub = [7, 0.001]
    pop_size = 3
    prob_size = len(lb)
    epochs = 100

    best_solution, best_fitness = sa_bbo(lb, ub, pop_size, prob_size, epochs, input_shape)

    val = max(1, best_solution[0].astype('int32'))
    dilation_rate = (val, val)
    learning_rate = best_solution[1]
    return fcn_model(input_shape, dilation_rate, learning_rate)

def generate_features(image):
    return extract_features(segment(histogram_equalize(normalize(denoise(transform(image))))))

class FeatureGenerator:
    def __init__(self):
        super().__init__()