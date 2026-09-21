import os
import site
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
REFERENCE_COSYVOICE_DIR = Path(os.environ.get("COSYVOICE_REFERENCE_DIR", ROOT_DIR))
WETEXT_MODEL_ID = "pengzhendong/wetext"

DEFAULT_MODELSCOPE_CACHE = REFERENCE_COSYVOICE_DIR / "cache"
DEFAULT_WETEXT_DIR = DEFAULT_MODELSCOPE_CACHE / "hub" / "pengzhendong" / "wetext"
DEFAULT_TRITON_CACHE_DIR = Path("/tmp/triton-cache")
DEFAULT_NUMBA_CACHE_DIR = Path("/tmp/numba-cache")
DEFAULT_MPLCONFIG_DIR = Path("/tmp/matplotlib-cache")


def disable_user_site_packages():
    if os.environ.get("COSYVOICE_ALLOW_USER_SITE") == "1":
        return

    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    try:
        user_sites = site.getusersitepackages()
    except AttributeError:
        return
    if isinstance(user_sites, str):
        user_sites = [user_sites]

    resolved_user_sites = {str(Path(path).resolve()) for path in user_sites}
    sys.path[:] = [
        path for path in sys.path
        if str(Path(path).resolve()) not in resolved_user_sites
    ]


def setup_python_paths():
    for path in (ROOT_DIR, ROOT_DIR / "third_party" / "Matcha-TTS"):
        path = str(path)
        if path not in sys.path:
            sys.path.insert(0, path)


def _set_path_env(name: str, path: str | os.PathLike):
    path = str(path)
    os.environ.setdefault(name, path)
    try:
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def setup_runtime_paths():
    disable_user_site_packages()
    setup_python_paths()
    _set_path_env("MODELSCOPE_CACHE", os.environ.get("COSYVOICE_MODELSCOPE_CACHE", DEFAULT_MODELSCOPE_CACHE))
    _set_path_env("TRITON_CACHE_DIR", os.environ.get("COSYVOICE_TRITON_CACHE_DIR", DEFAULT_TRITON_CACHE_DIR))
    _set_path_env("NUMBA_CACHE_DIR", os.environ.get("COSYVOICE_NUMBA_CACHE_DIR", DEFAULT_NUMBA_CACHE_DIR))
    _set_path_env("MPLCONFIGDIR", os.environ.get("COSYVOICE_MPLCONFIGDIR", DEFAULT_MPLCONFIG_DIR))


def get_wetext_dir() -> Path:
    return Path(os.environ.get("COSYVOICE_WETEXT_DIR", DEFAULT_WETEXT_DIR))


def has_wetext_files(wetext_dir: str | os.PathLike | None = None) -> bool:
    wetext_dir = Path(wetext_dir) if wetext_dir else get_wetext_dir()
    required_files = (
        wetext_dir / "zh" / "tn" / "tagger.fst",
        wetext_dir / "zh" / "tn" / "verbalizer.fst",
        wetext_dir / "en" / "tn" / "tagger.fst",
        wetext_dir / "en" / "tn" / "verbalizer.fst",
    )
    return all(path.exists() for path in required_files)


def wrap_snapshot_download(snapshot_download):
    if getattr(snapshot_download, "_cosyvoice_runtime_patched", False):
        return snapshot_download

    wetext_dir = get_wetext_dir()
    if not has_wetext_files(wetext_dir):
        return snapshot_download

    def wrapped(model_id, *args, **kwargs):
        if model_id == WETEXT_MODEL_ID:
            return str(wetext_dir)
        return snapshot_download(model_id, *args, **kwargs)

    wrapped._cosyvoice_runtime_patched = True
    return wrapped


def patch_wetext_snapshot_download():
    if not has_wetext_files():
        return

    try:
        import wetext.wetext as wetext_impl
    except ImportError:
        return

    wetext_impl.snapshot_download = wrap_snapshot_download(wetext_impl.snapshot_download)
