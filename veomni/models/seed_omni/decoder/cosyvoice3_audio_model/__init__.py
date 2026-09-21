from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("cosyvoice3_audio_decoder")
def register_cosyvoice3_audio_decoder_config():
    from .configuration_cosyvoice3_audio_model import CosyVoice3AudioDecoderConfig

    return CosyVoice3AudioDecoderConfig


@MODELING_REGISTRY.register("cosyvoice3_audio_decoder")
def register_cosyvoice3_audio_decoder_modeling(architecture: str):
    from .modeling_cosyvoice3_audio_model import CosyVoice3AudioDecoder

    return CosyVoice3AudioDecoder
