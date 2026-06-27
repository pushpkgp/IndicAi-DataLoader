import cv2
import numpy as np
from skimage.exposure import exposure
from skimage.restoration import denoise_wavelet

from app.service.feature.image.features import extract_features
from app.service.feature.image.segmentation import segment_image_result
from app.service.feature.preprocessor import logger


def transform(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(
        image,
        (256, 256),
        interpolation=cv2.INTER_AREA,
    )

    return image.astype("uint8")


def denoise(image: np.ndarray) -> np.ndarray:
    denoised_image = denoise_wavelet(
        image,
        channel_axis=None,
        rescale_sigma=True,
    )

    denoised_image = denoised_image * 255.0
    return np.clip(denoised_image, 0, 255).astype("uint8")


def normalize(denoised_image: np.ndarray) -> np.ndarray:
    return cv2.normalize(
        denoised_image,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )


def histogram_equalize(normalized_image: np.ndarray) -> np.ndarray:
    equalized_image = exposure.equalize_hist(normalized_image) * 255.0
    equalized_image = np.clip(equalized_image, 0, 255).astype("uint8")

    return cv2.cvtColor(
        equalized_image,
        cv2.COLOR_GRAY2BGR,
    )


def preprocess_image(image_path: str) -> np.ndarray:
    return histogram_equalize(
        normalize(
            denoise(
                transform(image_path)
            )
        )
    )


def segment(
    equalized_image: np.ndarray,
    segmentation_model,
    postprocess_params=None,
) -> np.ndarray:
    if segmentation_model is None:
        raise ValueError("Segmentation model is required for image feature extraction")

    logger.info("Running O-FCN segmentation inference")

    result = segment_image_result(
        image=equalized_image,
        model=segmentation_model,
        postprocess_params=postprocess_params,
    )

    logger.info("O-FCN segmentation complete")

    return result.segmented_image


def extract_image_features(
    image_path: str,
    segmentation_model,
    postprocess_params=None,
    deep_feature_extraction_model=None,
):
    logger.info("Extracting image features for %s", image_path)

    preprocessed_image = preprocess_image(image_path)

    segmented_image = segment(
        equalized_image=preprocessed_image,
        segmentation_model=segmentation_model,
        postprocess_params=postprocess_params,
    )

    features = extract_features(
        segmented_image,
        deep_feature_extraction_model,
    )

    logger.info("Feature extraction complete for %s", image_path)

    return features