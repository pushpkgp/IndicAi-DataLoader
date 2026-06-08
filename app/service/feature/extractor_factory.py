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
    def extractor(cls, modality, filepath, model):
        func = cls.get_extractor(modality)
        return func(filepath, model)

@Factory.register('image')
def image_feature_extractor(filepath, model):
    from app.service.feature.image.extractor import extract_image_features
    return extract_image_features(filepath, model)

@Factory.register('text')
def text_feature_extractor(filepath, model):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    # Replace this line with your text embedding logic
    return text[:512]  # Placeholder return

@Factory.register('audio')
def audio_feature_extractor(filepath, model):
    # Audio processing logic here
    return [0]  # Placeholder

@Factory.register('video')
def video_feature_extractor(filepath, model):
    # Video processing logic here
    return   # Placeholder