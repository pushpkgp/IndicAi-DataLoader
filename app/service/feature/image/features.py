from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import cv2
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input

_DENSENET_EXTRACTOR = None
_HOG = cv2.HOGDescriptor()

def safe_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x)
    if norm < eps or not np.isfinite(norm):
        return np.zeros_like(x, dtype=np.float32)
    return x / norm

def ensure_gray_uint8(image):
    if image is None:
        raise ValueError("Input image is None")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype != np.uint8:
        image = cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

    return image

def glcm(image):
    image = ensure_gray_uint8(image)

    co_matrix = graycomatrix(
        image,
        distances=[5],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )

    contrast = graycoprops(co_matrix, "contrast").ravel()
    correlation = graycoprops(co_matrix, "correlation").ravel()
    energy = graycoprops(co_matrix, "energy").ravel()
    homogeneity = graycoprops(co_matrix, "homogeneity").ravel()
    dissimilarity = graycoprops(co_matrix, "dissimilarity").ravel()

    return np.concatenate(
        [contrast, correlation, energy, homogeneity, dissimilarity]
    ).astype(np.float32)

def shape_features(image):
    image = ensure_gray_uint8(image)

    _, binary_image = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return np.zeros(3, dtype=np.float32)

    contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 1e-8:
        circularity = 0.0
    else:
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

    return np.array([area, perimeter, circularity], dtype=np.float32)

def lbp(image):
    image = ensure_gray_uint8(image)

    radius = 1
    n_points = 8 * radius

    lbp_feat = local_binary_pattern(
        image,
        P=n_points,
        R=radius,
        method="uniform",
    )

    hist, _ = np.histogram(
        lbp_feat.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2),
    )

    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-8

    return hist

def intensity_feature(image):
    image = ensure_gray_uint8(image)

    mean_intensity = np.mean(image)
    std_dev_intensity = np.std(image)

    histogram = cv2.calcHist(
        [image],
        [0],
        None,
        [8],
        [0, 256],
    ).flatten()

    histogram = histogram.astype(np.float32)
    histogram_norm = histogram / (histogram.sum() + 1e-8)

    entropy = -np.sum(
        histogram_norm * np.log2(histogram_norm + 1e-8)
    )

    sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    gradient_magnitude = np.mean(
        cv2.magnitude(sobelx, sobely)
    )

    image_blurred = cv2.GaussianBlur(image, (3, 3), 0)
    laplacian = np.mean(
        cv2.Laplacian(image_blurred, cv2.CV_32F)
    )

    # Resize for HOG descriptor compatibility.
    hog_img = cv2.resize(image, (64, 128))
    hog_features = _HOG.compute(hog_img)
    hog_mean = float(np.mean(hog_features)) if hog_features is not None else 0.0

    lbp_feat = lbp(image)

    intensity_feat = np.array(
        [
            mean_intensity,
            std_dev_intensity,
            entropy,
            gradient_magnitude,
            laplacian,
            hog_mean,
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        [histogram, lbp_feat, intensity_feat]
    ).astype(np.float32)

def get_deep_features(image, deep_feature_extraction_model):
    if image.ndim == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image

    img = cv2.resize(image_bgr, (224, 224))
    img_array = np.expand_dims(img.astype(np.float32), axis=0)
    img_array = preprocess_input(img_array)

    features = deep_feature_extraction_model(img_array, training=False).numpy()[0]
    return features.astype(np.float32)

def extract_features(image, deep_feature_extraction_model):
    image = ensure_gray_uint8(image)

    glcm_feature = glcm(image)
    shape_feature = shape_features(image)
    intensity_features = intensity_feature(image)
    deep_feature = get_deep_features(image, deep_feature_extraction_model)

    weight_glcm = 0.2
    weight_shape = 0.1
    weight_intensity = 0.3
    weight_deep = 0.4

    weighted_glcm = np.asarray(safe_normalize(glcm_feature) * weight_glcm).flatten()
    weighted_shape = np.asarray(safe_normalize(shape_feature) * weight_shape).flatten()
    weighted_intensity = np.asarray(safe_normalize(intensity_features) * weight_intensity).flatten()
    weighted_deep = np.asarray(safe_normalize(np.abs(deep_feature)) * weight_deep).flatten()

    feature = np.concatenate(
        [
            weighted_glcm,
            weighted_shape,
            weighted_intensity,
            weighted_deep,
        ]
    ).astype(np.float32)

    return feature