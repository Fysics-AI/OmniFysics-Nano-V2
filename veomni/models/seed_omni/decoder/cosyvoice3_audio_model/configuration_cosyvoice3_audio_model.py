from ..base import BaseDecoderConfigMixin


class CosyVoice3AudioDecoderConfig(BaseDecoderConfigMixin):
    model_type = "cosyvoice3_audio_decoder"

    def __init__(
        self,
        cosyvoice3_path: str = "/share/weights/CosyVoice3-0_5B",
        cosyvoice_repo_path: str = "/share/project/VeOmni/CosyVoice",
        llm_input_size: int = 896,
        llm_output_size: int = 896,
        speech_token_size: int = 6561,
        control_text: str = "You are a helpful assistant.<|endofprompt|>",
        projector_type: str = "mlp",
        projector_hidden_size: int = 2048,
        projector_depth: int = 2,
        foundation_tokenizer_path: str | None = None,
        cross_attention_num_heads: int = 14,
        cross_attention_dropout: float = 0.0,
        cross_attention_query_position_type: str = "sinusoidal",
        cross_attention_query_position_base: float = 10_000.0,
        cross_attention_query_position_scale: float | None = None,
        cross_attention_gate_init: float = 0.1,
        audio_loss_weight: float = 1.0,
        train_audio_output_projector: bool = True,
        train_cosyvoice3_lm: bool = True,
        freeze_cosyvoice3_non_lm: bool = True,
        load_pretrained: bool = True,
        use_dummy_lm: bool = False,
        dummy_vocab_size: int = 128,
        dummy_num_hidden_layers: int = 2,
        dummy_num_attention_heads: int = 2,
        dummy_num_key_value_heads: int = 1,
        dummy_intermediate_size: int = 64,
        output_size=None,
        add_projector: bool = True,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        obsolete_position_keys = {
            "cross_attention_max_positions",
            "cross_attention_position_type",
            "cross_attention_rope_theta",
        }.intersection(kwargs)
        if obsolete_position_keys:
            raise ValueError(
                f"Obsolete text-condition bridge position settings: {sorted(obsolete_position_keys)}. "
                "Use cross_attention_query_position_type='sinusoidal' and retrain the audio output projector and "
                "bridge."
            )
        if cross_attention_query_position_type != "sinusoidal":
            raise ValueError(
                "CosyVoice3 text-condition bridge only supports query-side dynamic sinusoidal positions, got "
                f"cross_attention_query_position_type={cross_attention_query_position_type!r}."
            )
        if cross_attention_query_position_base <= 0:
            raise ValueError(
                f"cross_attention_query_position_base must be positive, got {cross_attention_query_position_base}."
            )
        if cross_attention_query_position_scale is not None and cross_attention_query_position_scale <= 0:
            raise ValueError(
                "cross_attention_query_position_scale must be positive when set, got "
                f"{cross_attention_query_position_scale}."
            )

        super().__init__(
            output_size=output_size,
            add_projector=add_projector,
            initializer_range=initializer_range,
            architectures=kwargs.pop("architectures", ["CosyVoice3AudioDecoder"]),
            **kwargs,
        )
        self.cosyvoice3_path = cosyvoice3_path
        self.cosyvoice_repo_path = cosyvoice_repo_path
        self.llm_input_size = llm_input_size
        self.llm_output_size = llm_output_size
        self.speech_token_size = speech_token_size
        self.control_text = control_text
        self.projector_type = projector_type
        self.projector_hidden_size = projector_hidden_size
        self.projector_depth = projector_depth
        self.foundation_tokenizer_path = foundation_tokenizer_path
        self.cross_attention_num_heads = cross_attention_num_heads
        self.cross_attention_dropout = cross_attention_dropout
        self.cross_attention_query_position_type = cross_attention_query_position_type
        self.cross_attention_query_position_base = cross_attention_query_position_base
        self.cross_attention_query_position_scale = cross_attention_query_position_scale
        self.cross_attention_gate_init = cross_attention_gate_init
        self.audio_loss_weight = audio_loss_weight
        self.train_audio_output_projector = train_audio_output_projector
        self.train_cosyvoice3_lm = train_cosyvoice3_lm
        self.freeze_cosyvoice3_non_lm = freeze_cosyvoice3_non_lm
        self.load_pretrained = load_pretrained
        self.use_dummy_lm = use_dummy_lm
        self.dummy_vocab_size = dummy_vocab_size
        self.dummy_num_hidden_layers = dummy_num_hidden_layers
        self.dummy_num_attention_heads = dummy_num_attention_heads
        self.dummy_num_key_value_heads = dummy_num_key_value_heads
        self.dummy_intermediate_size = dummy_intermediate_size
