from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model

_DENSENET_EXTRACTOR = None

def get_densenet_extractor_model():
    global _DENSENET_EXTRACTOR

    if _DENSENET_EXTRACTOR is None:
        base_model = DenseNet121(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224,224,3)
        )

        _DENSENET_EXTRACTOR = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer("conv5_block16_concat").output,
        )

    return _DENSENET_EXTRACTOR