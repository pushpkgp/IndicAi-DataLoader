from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import cv2
import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model

def glcm(image):
    co_matrix = graycomatrix(image, [5], [0], 256, True, True)

    contrast = graycoprops(co_matrix, 'contrast').ravel()
    correlation = graycoprops(co_matrix, 'correlation').ravel()
    energy = graycoprops(co_matrix, 'energy').ravel()
    homogeneity = graycoprops(co_matrix, 'homogeneity').ravel()
    dissimilarity = graycoprops(co_matrix, 'dissimilarity').ravel()

    glcm_feat = np.concatenate([contrast, correlation, energy, homogeneity, dissimilarity])
    return glcm_feat

def shape_features(image):

    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate shape features for each contour
    for i, contour in enumerate(contours):
        # Calculate area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0.0:
            perimeter = 0.1
        # Calculate circularity
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        shape_feat = np.array([area, perimeter, circularity])
        return shape_feat
    return None


def lbp(image):
    radius = 1
    n_points = 8 * radius
    lbp_feat = local_binary_pattern(image, P=n_points, R=radius)

    hist, _ = np.histogram(lbp_feat.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def intensity_feature(image):
    # Compute mean intensity
    mean_intensity = np.mean(image)

    # Compute standard deviation of intensity
    std_dev_intensity = np.std(image)

    # Compute histogram
    histogram = cv2.calcHist([image], [0], None, [8], [0, 256])
    histogram = histogram.flatten()

    # Compute entropy
    histogram_normalized = histogram / np.sum(histogram)
    entropy = -np.sum(histogram_normalized * np.log2(histogram_normalized + 1e-8))

    # Compute gradient magnitude
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    gradient_magnitude = np.mean(gradient_magnitude)

    # Compute Laplacian of Gaussian
    image_blurred = cv2.GaussianBlur(image, (3, 3), 0)
    laplacian = cv2.Laplacian(image_blurred, cv2.CV_64F)
    laplacian = np.mean(laplacian)

    # Compute Histogram of Oriented Gradients (HOG)
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(image)
    hog_features = np.mean(hog_features)

    # Compute Local Binary Patterns (LBP)
    lbp_feat = lbp(image)
    intensity_feat = np.array([mean_intensity, std_dev_intensity, entropy, gradient_magnitude, laplacian, hog_features])
    intensity_feat = np.concatenate([histogram, lbp_feat, intensity_feat])
    return intensity_feat

def densenet121(image):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Load the pre-trained DenseNet121 model
    base_model = DenseNet121(weights='imagenet', include_top=False)

    # Define a new model with the desired output layer
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block16_concat').output)

    img = cv2.resize(image, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features from the image
    features = feature_extractor.predict(img_array)
    features = (np.mean(np.mean(features, axis=1), axis=1)).flatten()
    return features

def extract_features(image):
    glcm_feature = glcm(image)  # Texture feature

    shape_feature = shape_features(image)
    if shape_feature is None:
        shape_feature = np.zeros(3)

    intensity_features = intensity_feature(image)

    # deep learning based feature
    deep_feature = densenet121(image)

    # normalize the features
    glcm_feature_normalized = glcm_feature / np.linalg.norm(glcm_feature)
    shape_feature_normalized = shape_feature / np.linalg.norm(shape_feature)
    intensity_feature_normalized = intensity_features / np.linalg.norm(intensity_features)
    deep_feature_normalized = abs(deep_feature) / np.linalg.norm(abs(deep_feature))

    # Weighted Feature Fusion
    weight_glcm = 0.2
    weight_shape = 0.1
    weight_intensity = 0.3
    weight_deep = 0.4

    weighted_glcm = glcm_feature_normalized * weight_glcm
    weighted_shape = shape_feature_normalized * weight_shape
    weighted_intensity = intensity_feature_normalized * weight_intensity
    weighted_deep = deep_feature_normalized * weight_deep

    feature = np.concatenate([weighted_glcm, weighted_shape, weighted_intensity, weighted_deep])
    return feature
