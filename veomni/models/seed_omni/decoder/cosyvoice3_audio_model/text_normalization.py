import logging
import os
import re
import sys


logger = logging.getLogger(__name__)


class CosyVoiceTextNormalizer:
    """CosyVoice text normalization without loading audio ONNX models."""

    _think_block_pattern = re.compile(r"<think(?:\s[^>]*)?>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
    _unclosed_think_pattern = re.compile(r"<think(?:\s[^>]*)?>.*$", re.IGNORECASE | re.DOTALL)
    _heading_pattern = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
    _list_marker_pattern = re.compile(r"(?m)^\s*[-+*]\s+")

    @classmethod
    def clean_for_speech(cls, text: str) -> str:
        """Convert common LLM Markdown/control output into natural spoken text."""
        text = text.replace("\\<", "<").replace("\\>", ">")
        text = cls._think_block_pattern.sub(" ", text)
        text = cls._unclosed_think_pattern.sub(" ", text)

        text = re.sub(r"\\([`*_#\\])", r"\1", text)
        text = text.replace("**", "").replace("__", "").replace("`", "")

        text = cls._heading_pattern.sub("。", text)
        text = cls._list_marker_pattern.sub("，", text)
        text = re.sub(r"(?<=[\u3400-\u9fff。！？])\s*[-+*]\s*(?=[\u3400-\u9fff])", "，", text)
        text = re.sub(r"(?<!\\)#{1,6}(?=\s|\d|[\u3400-\u9fff])", "。", text)

        text = re.sub(r"[ \t]*\r?\n+[ \t]*", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def __init__(self, cosyvoice_repo_path: str) -> None:
        if cosyvoice_repo_path and os.path.isdir(cosyvoice_repo_path) and cosyvoice_repo_path not in sys.path:
            sys.path.insert(0, cosyvoice_repo_path)

        import inflect
        from cosyvoice.utils.frontend_utils import (
            contains_chinese,
            is_only_punctuation,
            remove_bracket,
            replace_blank,
            replace_corner_mark,
            spell_out_number,
        )
        from cosyvoice.utils.runtime_paths import patch_wetext_snapshot_download
        self.contains_chinese = contains_chinese
        self.is_only_punctuation = is_only_punctuation
        self.remove_bracket = remove_bracket
        self.replace_blank = replace_blank
        self.replace_corner_mark = replace_corner_mark
        self.spell_out_number = spell_out_number
        self.inflect_parser = inflect.engine()
        self.zh_tn_model = None
        self.en_tn_model = None

        try:
            patch_wetext_snapshot_download()
            from wetext import Normalizer

            self.zh_tn_model = Normalizer(remove_erhua=False)
            self.en_tn_model = Normalizer()
        except Exception as exc:
            logger.warning(
                "CosyVoice WeText normalization is unavailable; "
                "using deterministic punctuation/number cleanup only: %r",
                exc,
            )

    def normalize(self, text: str) -> str:
        text = self.clean_for_speech(text)
        if not text:
            raise ValueError("Speech source text is empty; cannot build a normalized CosyVoice query.")

        if "<|" not in text or "|>" not in text:
            if self.contains_chinese(text):
                if self.zh_tn_model is not None:
                    text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = self.replace_blank(text)
                text = self.replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = self.remove_bracket(text)
                text = re.sub(r"[，,、]+$", "。", text)
            else:
                if self.en_tn_model is not None:
                    text = self.en_tn_model.normalize(text)
                text = self.spell_out_number(text, self.inflect_parser)

        text = text.strip()
        if not text or self.is_only_punctuation(text):
            raise ValueError("CosyVoice text normalization produced empty or punctuation-only speech text.")
        return text
