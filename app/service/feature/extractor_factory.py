class Factory:
    _extractors = {}

    @classmethod
    def register(cls, modality):
        def inner(func):
            cls._extractors[modality] = func
            return func
        return inner

    @classmethod
    def get_extractor(cls, modality):
        if modality in cls._extractors:
            return cls._extractors[modality]
        raise ValueError(f"No extractor registered for modality: {modality}")

    @classmethod
    def extractor(cls, modality, filepath, model=None, postprocess_params=None, deep_feature_extraction_model=None):
        func = cls.get_extractor(modality)
        return func(filepath, model, postprocess_params, deep_feature_extraction_model)

@Factory.register("image")
def image_feature_extractor(filepath, segmentation_model=None, postprocess_params=None, deep_feature_extraction_model=None):
    from app.service.feature.image.extractor import extract_image_features
    return extract_image_features(filepath, segmentation_model, postprocess_params, deep_feature_extraction_model)

@Factory.register("text")
def text_feature_extractor(filepath, model=None, postprocess_params=None):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return text[:512]

@Factory.register("audio")
def audio_feature_extractor(filepath, model=None, postprocess_params=None):
    return [0]

@Factory.register("video")
def video_feature_extractor(filepath, model=None, postprocess_params=None):
    return [0]