import gc
import threading

from .loader import load_model_bundle


class LlmRuntime:
    def __init__(self):
        self._lock = threading.Lock()
        self._bundle = None
        self._signature = None

    def ensure_loaded(self, spec, logger=None):
        signature = spec.signature()
        with self._lock:
            if self._bundle is not None and self._signature == signature:
                if logger:
                    logger(f"runtime reuse model={spec.key}")
                    self._log_bundle_summary(self._bundle, logger, prefix="runtime_reuse")
                return self._bundle, False

            if self._bundle is not None:
                self._unload_locked(logger=logger)

            bundle = load_model_bundle(spec, logger=logger)
            self._bundle = bundle
            self._signature = signature
            if logger:
                logger(f"runtime loaded model={spec.key}")
                self._log_bundle_summary(bundle, logger, prefix="runtime_loaded")
            return bundle, True

    def unload(self, logger=None):
        with self._lock:
            return self._unload_locked(logger=logger)

    def is_loaded(self):
        with self._lock:
            return self._bundle is not None

    def _unload_locked(self, logger=None):
        if self._bundle is None:
            return False

        bundle = self._bundle
        self._bundle = None
        self._signature = None

        try:
            del bundle.model
        except Exception:
            pass

        try:
            del bundle.tokenizer
        except Exception:
            pass

        del bundle
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        if logger:
            logger("runtime unloaded current model")
        return True

    def _log_bundle_summary(self, bundle, logger, prefix):
        debug = getattr(bundle, "load_debug", {}) or {}
        logger(
            f"{prefix} "
            f"is_peft_model={debug.get('is_peft_model')} "
            f"active_adapter={debug.get('active_adapter')} "
            f"tokenizer_source={debug.get('tokenizer_source')} "
            f"chat_template_source={debug.get('chat_template_source')}"
        )


_RUNTIME = LlmRuntime()


def get_runtime():
    return _RUNTIME
