import re
import sys
import traceback
from dataclasses import dataclass, field, replace
from importlib import import_module
import importlib.util
from pathlib import Path

import gradio as gr

from modules import scripts, shared


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PACKAGE_NAME = "sd_forge_llm_prompt_gen_backend"
BACKEND_DIR = EXTENSION_ROOT / "backend"
BACKEND_INIT = BACKEND_DIR / "__init__.py"


def _load_backend_package():
    if BACKEND_PACKAGE_NAME in sys.modules:
        return sys.modules[BACKEND_PACKAGE_NAME]

    spec = importlib.util.spec_from_file_location(
        BACKEND_PACKAGE_NAME,
        BACKEND_INIT,
        submodule_search_locations=[str(BACKEND_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[BACKEND_PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


_load_backend_package()

compat_module = import_module(f"{BACKEND_PACKAGE_NAME}.compat")
generator_module = import_module(f"{BACKEND_PACKAGE_NAME}.generator")
loader_module = import_module(f"{BACKEND_PACKAGE_NAME}.loader")
prompt_builder_module = import_module(f"{BACKEND_PACKAGE_NAME}.prompt_builder")
registry_module = import_module(f"{BACKEND_PACKAGE_NAME}.registry")
runtime_module = import_module(f"{BACKEND_PACKAGE_NAME}.runtime")

GenerationError = generator_module.GenerationError
generate_prompt_candidates = generator_module.generate_prompt_candidates
apply_runtime_compat_shims = compat_module.apply_runtime_compat_shims
MissingDependencyError = loader_module.MissingDependencyError
ModelLoadError = loader_module.ModelLoadError
build_final_positive = prompt_builder_module.build_final_positive
build_final_positive_details = prompt_builder_module.build_final_positive_details
build_manual_positive = prompt_builder_module.build_manual_positive
summarize_text = prompt_builder_module.summarize_text
RegistryError = registry_module.RegistryError
get_generation_defaults = registry_module.get_generation_defaults
get_model_spec = registry_module.get_model_spec
get_ui_model_choices = registry_module.get_ui_model_choices
get_runtime = runtime_module.get_runtime


EXTENSION_ID = "sd-forge-llm-prompt-gen-yoiko"
LOG_PREFIX = f"[{EXTENSION_ID}]"
MAX_LOG_PROMPT_CHARS = 160
DEFAULT_LLM_MODEL_CHOICES = [
    "none",
    "qwen2.5-7b-instruct",
    "qwen3.5-4b",
    "qwen3.5-9b",
]
LLM_LOAD_MODE_CHOICES = [
    "keep_loaded",
    "load_then_unload_before_image_gen",
]
LLM_WEIGHT_MODE_CHOICES = [
    "auto",
    "4bit",
    "bf16_merge",
]


print(f"{LOG_PREFIX} extension script loaded")
apply_runtime_compat_shims(logger=lambda message: print(f"{LOG_PREFIX} {message}"))


@dataclass
class MergeDecision:
    llm_enabled: bool
    llm_model_name: str
    llm_load_mode: str
    llm_weight_mode: str
    llm_max_new_tokens: int
    llm_gen_prompt: str
    original_prompt: object
    final_prompt: object
    changed: bool
    llm_num_candidates: int = 1
    llm_preselected_candidate: str = ""
    llm_called: bool = False
    candidate_count: int = 0
    candidates: list[str] = field(default_factory=list)
    candidate_debug: list[dict] = field(default_factory=list)
    chosen_candidate: str = ""
    chosen_candidate_debug: dict = field(default_factory=dict)
    selected_positive: str = ""
    fallback_used: bool = False
    fallback_reason: str = ""
    llm_seed: int | None = None
    seed_mode: str = "random"
    interrupted: bool = False
    error: str = ""
    runtime_action: str = "not_requested"
    original_prompt_injected_to_llm: bool = False
    original_prompt_appended_after_llm: bool = False
    dedupe_applied: bool = False
    final_positive_before_dedupe: str = ""
    final_positive_after_dedupe: str = ""
    image_model_offloaded_to_ram: bool = False

    @property
    def gen_prompt_empty(self):
        return not bool(self.llm_gen_prompt)


def coerce_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def summarize_prompt(prompt_value):
    if isinstance(prompt_value, list):
        items = [coerce_text(item).strip() for item in prompt_value[:2]]
        summary = " | ".join(item for item in items if item)
        if len(prompt_value) > 2:
            summary = f"{summary} | ..."
    else:
        summary = coerce_text(prompt_value).strip()

    summary = re.sub(r"\s+", " ", summary)
    if len(summary) > MAX_LOG_PROMPT_CHARS:
        summary = summary[: MAX_LOG_PROMPT_CHARS - 3] + "..."
    return summary


def truncate_debug_text(value, max_chars=320):
    value = coerce_text(value).replace("\r", "")
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > max_chars:
        return value[: max_chars - 3] + "..."
    return value


def normalize_load_mode(value):
    if value in LLM_LOAD_MODE_CHOICES:
        return value
    return LLM_LOAD_MODE_CHOICES[0]


def normalize_weight_mode(value):
    normalized = coerce_text(value).strip().lower()
    if normalized in LLM_WEIGHT_MODE_CHOICES:
        return normalized
    if normalized in {"bf16", "merge", "merged"}:
        return "bf16_merge"
    return LLM_WEIGHT_MODE_CHOICES[0]


def normalize_max_new_tokens(value):
    try:
        return max(1, min(225, int(value)))
    except Exception:
        return 128


LLM_NUM_CANDIDATES_MIN = 1
LLM_NUM_CANDIDATES_MAX = 4


def normalize_num_candidates(value):
    try:
        return max(LLM_NUM_CANDIDATES_MIN, min(LLM_NUM_CANDIDATES_MAX, int(value)))
    except Exception:
        return 1


def build_initial_decision(
    llm_enabled,
    llm_model_name,
    llm_load_mode,
    llm_weight_mode,
    llm_max_new_tokens,
    llm_gen_prompt,
    original_prompt,
    llm_num_candidates=1,
    llm_preselected_candidate="",
):
    llm_model_name = coerce_text(llm_model_name).strip() or "none"
    llm_load_mode = normalize_load_mode(llm_load_mode)
    llm_weight_mode = normalize_weight_mode(llm_weight_mode)
    llm_max_new_tokens = normalize_max_new_tokens(llm_max_new_tokens)
    llm_num_candidates = normalize_num_candidates(llm_num_candidates)
    llm_gen_prompt = coerce_text(llm_gen_prompt).strip()
    llm_preselected_candidate = coerce_text(llm_preselected_candidate)
    original_prompt = coerce_text(original_prompt)

    if not llm_enabled:
        return MergeDecision(
            llm_enabled=False,
            llm_model_name=llm_model_name,
            llm_load_mode=llm_load_mode,
            llm_weight_mode=llm_weight_mode,
            llm_max_new_tokens=llm_max_new_tokens,
            llm_gen_prompt=llm_gen_prompt,
            original_prompt=original_prompt,
            final_prompt=original_prompt,
            changed=False,
            llm_num_candidates=llm_num_candidates,
            llm_preselected_candidate=llm_preselected_candidate,
        )

    if llm_model_name == "none":
        final_prompt = build_manual_positive(llm_gen_prompt, original_prompt)
        return MergeDecision(
            llm_enabled=True,
            llm_model_name=llm_model_name,
            llm_load_mode=llm_load_mode,
            llm_weight_mode=llm_weight_mode,
            llm_max_new_tokens=llm_max_new_tokens,
            llm_gen_prompt=llm_gen_prompt,
            original_prompt=original_prompt,
            final_prompt=final_prompt,
            changed=final_prompt != original_prompt,
            runtime_action="manual_prepend",
            llm_num_candidates=llm_num_candidates,
            llm_preselected_candidate=llm_preselected_candidate,
        )

    if not llm_gen_prompt:
        return MergeDecision(
            llm_enabled=True,
            llm_model_name=llm_model_name,
            llm_load_mode=llm_load_mode,
            llm_weight_mode=llm_weight_mode,
            llm_max_new_tokens=llm_max_new_tokens,
            llm_gen_prompt=llm_gen_prompt,
            original_prompt=original_prompt,
            final_prompt=original_prompt,
            changed=False,
            runtime_action="model_selected_but_gen_prompt_empty",
            llm_num_candidates=llm_num_candidates,
            llm_preselected_candidate=llm_preselected_candidate,
        )

    return MergeDecision(
        llm_enabled=True,
        llm_model_name=llm_model_name,
        llm_load_mode=llm_load_mode,
        llm_weight_mode=llm_weight_mode,
        llm_max_new_tokens=llm_max_new_tokens,
        llm_gen_prompt=llm_gen_prompt,
        original_prompt=original_prompt,
        final_prompt=original_prompt,
        changed=False,
        runtime_action=llm_load_mode,
        llm_num_candidates=llm_num_candidates,
        llm_preselected_candidate=llm_preselected_candidate,
    )


class ForgeLlmPromptGenScript(scripts.Script):
    sorting_priority = 15
    create_group = False

    def title(self):
        return "LLM Prompt Gen (Yoiko)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab_name = "img2img" if is_img2img else "txt2img"
        model_choices = get_ui_model_choices(DEFAULT_LLM_MODEL_CHOICES)

        with gr.Group(
            elem_id=f"{tab_name}_llm_prompt_gen_mount",
            elem_classes=["llm-prompt-gen-mount"],
        ):
            with gr.Row(elem_classes=["llm-prompt-gen-controls-row"]):
                llm_enabled = gr.Checkbox(
                    label="Enable LLM Prompt Gen",
                    value=False,
                    elem_id=f"{tab_name}_llm_prompt_gen_enabled",
                    elem_classes=["llm-prompt-gen-enabled"],
                )
                llm_model_name = gr.Dropdown(
                    label="LLM Model",
                    choices=model_choices,
                    value="none",
                    elem_id=f"{tab_name}_llm_model_name",
                    elem_classes=["llm-prompt-gen-model"],
                )
                llm_load_mode = gr.Dropdown(
                    label="LLM Load Mode",
                    choices=LLM_LOAD_MODE_CHOICES,
                    value="keep_loaded",
                    elem_id=f"{tab_name}_llm_load_mode",
                    elem_classes=["llm-prompt-gen-load-mode"],
                )
                llm_weight_mode = gr.Dropdown(
                    label="LLM Weight Mode",
                    choices=LLM_WEIGHT_MODE_CHOICES,
                    value="auto",
                    elem_id=f"{tab_name}_llm_weight_mode",
                    elem_classes=["llm-prompt-gen-weight-mode"],
                )
            llm_max_new_tokens = gr.Slider(
                label="LLM Max New Tokens",
                minimum=1,
                maximum=225,
                step=1,
                value=128,
                elem_id=f"{tab_name}_llm_max_new_tokens",
                elem_classes=["llm-prompt-gen-max-new-tokens"],
            )
            llm_num_candidates = gr.Slider(
                label="LLM Candidate Count",
                minimum=LLM_NUM_CANDIDATES_MIN,
                maximum=LLM_NUM_CANDIDATES_MAX,
                step=1,
                value=1,
                elem_id=f"{tab_name}_llm_num_candidates",
                elem_classes=["llm-prompt-gen-num-candidates"],
            )
            llm_gen_prompt = gr.Textbox(
                label="Gen Prompt",
                value="",
                lines=3,
                placeholder="Primary LLM input. Built-in Prompt remains in the final positive prompt.",
                elem_id=f"{tab_name}_llm_prompt_gen_text",
                elem_classes=["llm-prompt-gen-textbox"],
            )
            with gr.Row(elem_classes=["llm-prompt-gen-preview-row"]):
                llm_preview_button = gr.Button(
                    value="Preview Candidates",
                    elem_id=f"{tab_name}_llm_preview_button",
                    elem_classes=["llm-prompt-gen-preview-button"],
                )
                llm_preview_status = gr.Markdown(
                    value="",
                    elem_id=f"{tab_name}_llm_preview_status",
                    elem_classes=["llm-prompt-gen-preview-status"],
                )
            llm_candidate_radio = gr.Radio(
                label="Candidate (select one, or leave empty for auto-pick)",
                choices=[],
                value=None,
                interactive=True,
                elem_id=f"{tab_name}_llm_candidate_radio",
                elem_classes=["llm-prompt-gen-candidate-radio"],
            )
            llm_selected_candidate = gr.Textbox(
                value="",
                visible=False,
                elem_id=f"{tab_name}_llm_selected_candidate",
                elem_classes=["llm-prompt-gen-selected-candidate"],
            )

        def _on_preview_click(
            enabled,
            model_name,
            load_mode,
            weight_mode,
            max_new_tokens,
            num_candidates,
            gen_prompt,
        ):
            try:
                preview_decision = self._preview_candidates(
                    enabled,
                    model_name,
                    load_mode,
                    weight_mode,
                    max_new_tokens,
                    num_candidates,
                    gen_prompt,
                )
            except Exception as exc:
                self._log(f"preview_candidates failed: {exc}")
                traceback.print_exc()
                return (
                    gr.update(choices=[], value=None),
                    "",
                    f"preview failed: {exc}",
                )
            choices = list(preview_decision.candidates or [])
            initial = choices[0] if choices else None
            if not preview_decision.llm_enabled:
                status = "LLM is disabled — enable it to preview."
            elif preview_decision.llm_model_name == "none":
                status = "No model selected."
            elif not choices:
                status = preview_decision.error or preview_decision.fallback_reason or "No candidates generated."
            else:
                status = f"{len(choices)} candidate(s) generated."
            return (
                gr.update(choices=choices, value=initial),
                initial or "",
                status,
            )

        llm_preview_button.click(
            fn=_on_preview_click,
            inputs=[
                llm_enabled,
                llm_model_name,
                llm_load_mode,
                llm_weight_mode,
                llm_max_new_tokens,
                llm_num_candidates,
                llm_gen_prompt,
            ],
            outputs=[
                llm_candidate_radio,
                llm_selected_candidate,
                llm_preview_status,
            ],
        )

        def _on_radio_change(value):
            return coerce_text(value)

        llm_candidate_radio.change(
            fn=_on_radio_change,
            inputs=[llm_candidate_radio],
            outputs=[llm_selected_candidate],
        )

        return [
            llm_enabled,
            llm_model_name,
            llm_load_mode,
            llm_weight_mode,
            llm_max_new_tokens,
            llm_num_candidates,
            llm_gen_prompt,
            llm_selected_candidate,
        ]

    def _log(self, message):
        print(f"{LOG_PREFIX} {message}")

    def _is_interrupted(self):
        state = getattr(shared, "state", None)
        if state is None:
            return False
        return bool(getattr(state, "interrupted", False) or getattr(state, "stopping_generation", False))

    def _apply_weight_mode_to_spec(self, spec, weight_mode):
        weight_mode = normalize_weight_mode(weight_mode)
        spec = replace(spec, runtime_weight_mode=weight_mode)
        if weight_mode == "4bit":
            return replace(
                spec,
                load_in_4bit=True,
                merge_lora_for_inference=False,
            )
        if weight_mode == "bf16_merge":
            return replace(
                spec,
                load_in_4bit=False,
                merge_lora_for_inference=True,
                torch_dtype=spec.torch_dtype or "bfloat16",
            )
        return spec

    def _offload_current_image_model_to_ram(self):
        try:
            from backend import memory_management
            from modules import sd_models
        except Exception as exc:
            self._log(f"image_model_offload_unavailable error={exc}")
            return False

        loaded_models = getattr(memory_management, "current_loaded_models", None) or []
        loaded_count = len(loaded_models)
        if loaded_count <= 0:
            self._log("image_model_offload skipped=no_loaded_models")
            return False

        sd_models.unload_model_weights()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        self._log(
            f"image_model_offload_to_ram applied=True loaded_model_count={loaded_count} "
            "target=cpu_ram"
        )
        return True

    def _generate_with_llm(self, decision, negative_prompt, *, offload_image_model=False):
        runtime = get_runtime()

        _unload_state = {"done": False}

        def _unload_once():
            if _unload_state["done"]:
                return False
            _unload_state["done"] = True
            try:
                return runtime.unload(logger=self._log)
            except Exception as exc:
                self._log(f"unload failed: {exc}")
                return False

        if self._is_interrupted():
            decision.interrupted = True
            decision.error = "interrupted before llm call"
            decision.runtime_action = "skipped_due_to_interrupt"
            self._log("interrupt detected before llm generation; falling back to original prompt")
            return decision

        try:
            spec = get_model_spec(decision.llm_model_name)
            spec = self._apply_weight_mode_to_spec(spec, decision.llm_weight_mode)
            generation_defaults = dict(get_generation_defaults())
            generation_defaults["max_new_tokens"] = decision.llm_max_new_tokens
            aggressive_cleanup = decision.llm_load_mode != "keep_loaded"
            if (
                offload_image_model
                and decision.llm_load_mode == "load_then_unload_before_image_gen"
            ):
                decision.image_model_offloaded_to_ram = self._offload_current_image_model_to_ram()
            self._log(
                "llm_load_config "
                f"base_model_name_or_path={spec.base_model_name_or_path} "
                f"adapter_path={spec.adapter_path} "
                f"tokenizer_name_or_path={spec.tokenizer_name_or_path or spec.base_model_name_or_path} "
                f"weight_mode={decision.llm_weight_mode} "
                f"tokenizer_source={spec.tokenizer_source} "
                f"chat_template_source={spec.chat_template_source} "
                f"quantization={'int4' if spec.load_in_4bit else 'none'} "
                f"bnb_4bit_quant_type={spec.bnb_4bit_quant_type} "
                f"bnb_4bit_compute_dtype={spec.bnb_4bit_compute_dtype} "
                f"bnb_4bit_use_double_quant={bool(spec.use_double_quant)} "
                f"input_template_mode={generation_defaults.get('input_template_mode', 'simple_chat_template')} "
                f"max_new_tokens={generation_defaults.get('max_new_tokens', 128)} "
                f"cache_implementation={generation_defaults.get('cache_implementation', 'dynamic')} "
                f"use_cache={bool(generation_defaults.get('use_cache', True))}"
            )

            bundle, was_loaded = runtime.ensure_loaded(spec, logger=self._log)
            decision.runtime_action = "reused_loaded_model" if not was_loaded else "loaded_model"

            result = generate_prompt_candidates(
                bundle=bundle,
                gen_prompt=decision.llm_gen_prompt,
                original_prompt=decision.original_prompt,
                negative_prompt=negative_prompt,
                generation_defaults=generation_defaults,
                interrupt_checker=self._is_interrupted,
                logger=self._log,
                aggressive_cleanup=aggressive_cleanup,
                num_candidates=decision.llm_num_candidates,
            )

            decision.llm_called = result.llm_called
            decision.candidates = result.candidates
            decision.candidate_debug = result.candidate_debug
            decision.candidate_count = len(result.candidates)
            decision.chosen_candidate = result.chosen_candidate
            decision.chosen_candidate_debug = result.chosen_candidate_debug
            decision.selected_positive = result.selected_positive
            decision.fallback_used = result.fallback_used
            decision.fallback_reason = result.fallback_reason
            decision.llm_seed = result.llm_seed
            decision.seed_mode = result.seed_mode
            decision.interrupted = result.interrupted
            decision.original_prompt_injected_to_llm = bool(
                result.input_debug.get("original_prompt_injected_to_llm", False)
            )

            preselected = coerce_text(decision.llm_preselected_candidate).strip()
            if preselected and result.candidates and preselected in result.candidates:
                decision.chosen_candidate = preselected
                for entry in result.candidate_debug:
                    if entry.get("cleaned_output") == preselected:
                        decision.chosen_candidate_debug = entry
                        break
                decision.selected_positive = preselected
                decision.fallback_used = False
                decision.fallback_reason = ""
                self._log(f"using user-preselected candidate: {summarize_text(preselected, 70)}")

            final_positive_details = build_final_positive_details(
                decision.selected_positive,
                decision.original_prompt,
                append_original=True,
            )
            decision.original_prompt_appended_after_llm = final_positive_details["original_prompt_appended_after_llm"]
            decision.dedupe_applied = final_positive_details["dedupe_applied"]
            decision.final_positive_before_dedupe = final_positive_details["final_positive_before_dedupe"]
            decision.final_positive_after_dedupe = final_positive_details["final_positive_after_dedupe"]
            decision.final_prompt = final_positive_details["final_positive_after_dedupe"]
            decision.changed = decision.final_prompt != decision.original_prompt

            if decision.llm_load_mode == "load_then_unload_before_image_gen":
                _unload_once()
                if decision.image_model_offloaded_to_ram:
                    decision.runtime_action = "load_then_unload_before_image_gen_with_sd_ram_offload"
                else:
                    decision.runtime_action = "load_then_unload_before_image_gen"
            else:
                decision.runtime_action = "keep_loaded"

            return decision
        except (RegistryError, MissingDependencyError, ModelLoadError) as exc:
            decision.error = str(exc)
            decision.final_prompt = decision.original_prompt
            decision.changed = False
            self._log(f"llm setup failed: {exc}; falling back to original prompt")
        except GenerationError as exc:
            decision.error = str(exc)
            decision.final_prompt = decision.original_prompt
            decision.changed = False
            if exc.is_oom:
                self._log(f"llm generation OOM: {exc}; falling back to original prompt")
                _unload_once()
            else:
                self._log(f"llm generation failed: {exc}; falling back to original prompt")
        except Exception as exc:
            decision.error = str(exc)
            decision.final_prompt = decision.original_prompt
            decision.changed = False
            self._log(f"unexpected llm error: {exc}; falling back to original prompt")
            traceback.print_exc()
        finally:
            if decision.llm_load_mode == "load_then_unload_before_image_gen":
                _unload_once()

        return decision

    def _preview_candidates(
        self,
        llm_enabled,
        llm_model_name,
        llm_load_mode,
        llm_weight_mode,
        llm_max_new_tokens,
        llm_num_candidates,
        llm_gen_prompt,
    ):
        original_prompt = coerce_text(llm_gen_prompt)
        decision = build_initial_decision(
            llm_enabled,
            llm_model_name,
            llm_load_mode,
            llm_weight_mode,
            llm_max_new_tokens,
            llm_gen_prompt,
            original_prompt,
            llm_num_candidates=llm_num_candidates,
            llm_preselected_candidate="",
        )
        if not decision.llm_enabled or decision.llm_model_name == "none" or decision.gen_prompt_empty:
            return decision
        return self._generate_with_llm(decision, negative_prompt="", offload_image_model=False)

    def before_process(
        self,
        p,
        llm_enabled,
        llm_model_name,
        llm_load_mode,
        llm_weight_mode,
        llm_max_new_tokens,
        llm_num_candidates,
        llm_gen_prompt,
        llm_selected_candidate,
    ):
        try:
            original_prompt = getattr(p, "prompt", "")
            negative_prompt = getattr(p, "negative_prompt", "")

            decision = build_initial_decision(
                llm_enabled,
                llm_model_name,
                llm_load_mode,
                llm_weight_mode,
                llm_max_new_tokens,
                llm_gen_prompt,
                original_prompt,
                llm_num_candidates=llm_num_candidates,
                llm_preselected_candidate=llm_selected_candidate,
            )

            if not decision.llm_enabled:
                runtime = get_runtime()
                unloaded = runtime.unload(logger=self._log)
                decision.runtime_action = "disabled_and_unloaded_model" if unloaded else "disabled_no_loaded_model"
            elif decision.llm_model_name != "none" and not decision.gen_prompt_empty:
                decision = self._generate_with_llm(
                    decision,
                    negative_prompt,
                    offload_image_model=True,
                )

            setattr(p, "_llm_prompt_gen_decision", decision)
            setattr(p, "_llm_prompt_gen_candidates", list(decision.candidates))

            if decision.changed:
                p.prompt = decision.final_prompt
        except Exception as exc:
            self._log(f"prompt merge failed in before_process: {exc}")
            traceback.print_exc()

    def process(
        self,
        p,
        llm_enabled,
        llm_model_name,
        llm_load_mode,
        llm_weight_mode,
        llm_max_new_tokens,
        llm_num_candidates,
        llm_gen_prompt,
        llm_selected_candidate,
    ):
        try:
            decision = getattr(p, "_llm_prompt_gen_decision", None)
            if decision is None:
                decision = build_initial_decision(
                    llm_enabled,
                    llm_model_name,
                    llm_load_mode,
                    llm_weight_mode,
                    llm_max_new_tokens,
                    llm_gen_prompt,
                    getattr(p, "prompt", ""),
                    llm_num_candidates=llm_num_candidates,
                    llm_preselected_candidate=llm_selected_candidate,
                )

            if not hasattr(p, "extra_generation_params") or p.extra_generation_params is None:
                p.extra_generation_params = {}

            if not decision.llm_enabled:
                p.extra_generation_params["LLM Prompt Gen"] = "off"
            elif decision.llm_model_name == "none":
                p.extra_generation_params["LLM Prompt Gen"] = "manual prepend"
            elif decision.gen_prompt_empty:
                p.extra_generation_params["LLM Prompt Gen"] = "model selected (gen prompt empty)"
            elif decision.llm_called:
                p.extra_generation_params["LLM Prompt Gen"] = "llm generated"
            else:
                p.extra_generation_params["LLM Prompt Gen"] = "llm fallback"

            p.extra_generation_params["LLM Model"] = decision.llm_model_name
            p.extra_generation_params["LLM Load Mode"] = decision.llm_load_mode
            p.extra_generation_params["LLM Weight Mode"] = decision.llm_weight_mode
            p.extra_generation_params["LLM Max New Tokens"] = decision.llm_max_new_tokens
            p.extra_generation_params["LLM Num Candidates"] = decision.llm_num_candidates
            p.extra_generation_params["LLM Candidate Count"] = decision.candidate_count
            p.extra_generation_params["LLM Image Model Offloaded To RAM"] = decision.image_model_offloaded_to_ram
            if decision.llm_preselected_candidate:
                p.extra_generation_params["LLM Candidate Selection"] = "user_selected"
            elif decision.candidate_count > 1:
                p.extra_generation_params["LLM Candidate Selection"] = "auto_pick"
            p.extra_generation_params["LLM Called"] = decision.llm_called
            p.extra_generation_params["LLM Fallback Used"] = decision.fallback_used
            p.extra_generation_params["LLM Seed Mode"] = decision.seed_mode
            if decision.llm_seed is not None:
                p.extra_generation_params["LLM Seed"] = decision.llm_seed

            if decision.chosen_candidate:
                p.extra_generation_params["LLM Chosen Candidate"] = summarize_text(decision.chosen_candidate, 60)
            if decision.selected_positive:
                p.extra_generation_params["LLM Selected Positive"] = summarize_text(decision.selected_positive, 60)
            p.extra_generation_params["LLM Prompt Injected To LLM"] = decision.original_prompt_injected_to_llm
            p.extra_generation_params["LLM Prompt Appended After LLM"] = decision.original_prompt_appended_after_llm
            p.extra_generation_params["LLM Dedupe Applied"] = decision.dedupe_applied
            self._log(
                f"enabled={decision.llm_enabled} "
                f"model={decision.llm_model_name} "
                f"load_mode={decision.llm_load_mode} "
                f"weight_mode={decision.llm_weight_mode} "
                f"max_new_tokens={decision.llm_max_new_tokens} "
                f"num_candidates={decision.llm_num_candidates} "
                f"candidate_count={decision.candidate_count} "
                f"user_selected={bool(decision.llm_preselected_candidate)} "
                f"image_model_offloaded_to_ram={decision.image_model_offloaded_to_ram} "
                f"gen_prompt_empty={decision.gen_prompt_empty} "
                f"seed_mode={decision.seed_mode} "
                f"llm_seed={decision.llm_seed} "
                f"llm_called={decision.llm_called} "
                f"runtime_action={decision.runtime_action}"
            )

            if decision.candidate_debug:
                item = decision.candidate_debug[0]
                self._log(
                    f"raw_model_output={truncate_debug_text(item.get('raw_model_output'), 240)}"
                )
                self._log(
                    f"cleaned_output={truncate_debug_text(item.get('cleaned_output'), 240)}"
                )

            if decision.chosen_candidate:
                self._log(f"chosen_candidate={summarize_text(decision.chosen_candidate, 70)}")
            else:
                self._log("chosen_candidate=<empty>; using original prompt fallback")

            if decision.chosen_candidate_debug:
                self._log(
                    "chosen_candidate_debug "
                    f"raw_model_output={truncate_debug_text(decision.chosen_candidate_debug.get('raw_model_output'), 320)} "
                    f"normalized_output={truncate_debug_text(decision.chosen_candidate_debug.get('normalized_output'), 320)} "
                    f"cleaned_output={truncate_debug_text(decision.chosen_candidate_debug.get('cleaned_output'), 320)}"
                )

            self._log(f"original_prompt_injected_to_llm={decision.original_prompt_injected_to_llm}")
            self._log(f"original_prompt_appended_after_llm={decision.original_prompt_appended_after_llm}")
            self._log(f"dedupe_applied={decision.dedupe_applied}")
            self._log(f"fallback_used={decision.fallback_used}")
            if decision.fallback_reason:
                self._log(f"fallback_reason={decision.fallback_reason}")

            if decision.error:
                self._log(f"error_reason={summarize_text(decision.error, 100)}")

            self._log(f"original_prompt={summarize_prompt(decision.original_prompt)}")
            self._log(f"final_positive_before_dedupe={summarize_prompt(decision.final_positive_before_dedupe)}")
            self._log(f"final_positive_after_dedupe={summarize_prompt(decision.final_positive_after_dedupe)}")
            self._log(f"final_positive_preview={summarize_prompt(decision.final_prompt)}")
            self._log(f"final_negative_preview={summarize_prompt(getattr(p, 'negative_prompt', ''))}")
        except Exception as exc:
            self._log(f"prompt logging failed in process: {exc}")
            traceback.print_exc()
