import gc
import threading
import traceback

from .loader import MissingDependencyError, ModelLoadError, load_model_bundle


def _prepare_cuda_workspaces(spec, logger=None):
    """Keep persistent BLAS scratch space out of the model's large allocation."""
    if getattr(spec, "backend", "transformers") != "transformers":
        return
    device_map = getattr(spec, "device_map", "auto")
    if device_map is None or device_map == "cpu":
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        if isinstance(device_map, str) and device_map in {"auto", "balanced", "balanced_low_0", "sequential"}:
            devices = set(range(torch.cuda.device_count()))
        else:
            devices = set()
            placements = device_map.values() if isinstance(device_map, dict) else [device_map]
            for placement in placements:
                try:
                    device = torch.device("cuda", placement) if isinstance(placement, int) else torch.device(placement)
                except (TypeError, ValueError, RuntimeError):
                    continue
                if device.type == "cuda":
                    devices.add(torch.cuda.current_device() if device.index is None else device.index)
        for device in sorted(devices):
            with torch.cuda.device(device):
                # HF warms its allocator with a model-sized segment. A small
                # persistent cuBLAS workspace allocated inside that segment can
                # keep gigabytes reserved after the model has been collected.
                # A separate pool also protects a reused model on a new worker
                # thread. Do not clear global BLAS caches shared with Forge.
                with torch.cuda.use_mem_pool(torch.cuda.MemPool()):
                    torch.cuda.current_blas_handle()
    except Exception as exc:
        # This is an allocator optimization, not a model-loading requirement.
        # Older Torch builds or other allocator backends may lack pool support.
        if logger:
            logger(f"runtime CUDA workspace preparation skipped: {exc}")

def _clear_cuda_cache():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

class LlmRuntime:
    def __init__(self):
        self._lock = threading.Lock()
        self._bundle = None
        self._signature = None

    def ensure_loaded(self, spec, logger=None):
        signature = spec.signature()
        with self._lock:
            if self._bundle is not None and self._signature == signature:
                _prepare_cuda_workspaces(spec, logger=logger)
                if logger:
                    logger(f"runtime reuse model={spec.key}")
                    self._log_bundle_summary(self._bundle, logger, prefix="runtime_reuse")
                return self._bundle, False

            if self._bundle is not None:
                self._unload_locked(logger=logger)

            _prepare_cuda_workspaces(spec, logger=logger)
            load_error = None
            try:
                bundle = load_model_bundle(spec, logger=logger)
            except Exception as exc:
                # Keep diagnostics without retaining partial models through traceback frames.
                error_type = MissingDependencyError if isinstance(exc, MissingDependencyError) else ModelLoadError
                load_error = error_type(str(exc))
                load_error.load_traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            if load_error is not None:
                self._bundle = None
                self._signature = None
                _clear_cuda_cache()
                raise load_error from None
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
            close = getattr(bundle.model, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

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
