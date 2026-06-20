import cv2
from skimage.exposure import exposure
from skimage.restoration import denoise_wavelet

from app.service.feature.image.features import extract_features
from app.service.feature.image.segmentation import segment_image
from app.service.feature.preprocessor import logger

def transform(image_path: str):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    image = cv2.resize(image, (256, 256))
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def denoise(image):
    denoised_image = denoise_wavelet(image) * 255
    return denoised_image.astype("uint8")

def normalize(denoised_image):
    return cv2.normalize(
        denoised_image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

def histogram_equalize(normalized_image):
    equalized_image = exposure.equalize_hist(normalized_image) * 255
    return cv2.cvtColor(equalized_image.astype("uint8"), cv2.COLOR_GRAY2BGR)

def segment(equalized_image, segmentation_model, postprocess_params=None):
    if segmentation_model is None:
        raise ValueError("Segmentation model is required for image feature extraction")

    logger.info("Running O-FCN segmentation inference")

    segmented_image = segment_image(
        image=equalized_image,
        model=segmentation_model,
        postprocess_params=postprocess_params,
    )

    logger.info("O-FCN segmentation complete")
    return segmented_image

def extract_image_features(image_path: str, segmentation_model, postprocess_params=None, deep_feature_extraction_model=None):
    logger.info("Extracting image features for %s", image_path)
    features = extract_features(segment(
        equalized_image=histogram_equalize(normalize(denoise(transform(image_path)))),
        segmentation_model=segmentation_model,
        postprocess_params=postprocess_params,
    ), deep_feature_extraction_model)

    logger.info("Feature extraction complete for %s", image_path)
    return features