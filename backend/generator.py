import gc
import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from .prompt_builder import (
    SYSTEM_PROMPT,
    build_messages,
    build_manual_positive,
    build_plain_text_input,
    build_simple_chat_messages,
    clean_generated_positive,
    is_generated_prompt_strong_enough,
    normalize_model_output,
)


class GenerationError(Exception):
    def __init__(self, message, is_oom=False):
        super().__init__(message)
        self.is_oom = is_oom


@dataclass
class PreparedInput:
    encoded: dict
    prompt_text: str
    tokenized_input_length: int
    decoded_prompt_preview: str
    input_mode: str
    chat_template_used: bool
    system_prompt_preview: str
    user_prompt_preview: str
    empty_think_block_stripped: bool = False


@dataclass
class GenerationResult:
    llm_called: bool
    candidates: list[str] = field(default_factory=list)
    candidate_debug: list[dict] = field(default_factory=list)
    chosen_candidate: str = ""
    chosen_candidate_debug: dict = field(default_factory=dict)
    selected_positive: str = ""
    fallback_used: bool = False
    fallback_reason: str = ""
    llm_seed: int | None = None
    seed_mode: str = "random"
    input_debug: dict = field(default_factory=dict)
    interrupted: bool = False
    generate_seconds: float = 0.0


def _import_torch():
    import torch

    return torch


def _get_model_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _safe_log(logger, message):
    if logger:
        logger(message)


def _truncate(value, max_chars=400):
    value = "" if value is None else str(value)
    value = value.replace("\r", "")
    if len(value) > max_chars:
        return value[: max_chars - 3] + "..."
    return value




def _build_input_payload(input_template_mode, gen_prompt, original_prompt, negative_prompt):
    mode = (input_template_mode or "simple_chat_template").strip().lower()
    if mode == "forge_prompt_builder":
        messages = build_messages(gen_prompt, "", negative_prompt)
        return {
            "messages": messages,
            "plain_text": None,
            "input_template_mode": "forge_prompt_builder",
            "original_prompt_injected_to_llm": False,
        }

    messages = build_simple_chat_messages(gen_prompt, "", negative_prompt)
    return {
        "messages": messages,
        "plain_text": None,
        "input_template_mode": "simple_chat_template",
        "original_prompt_injected_to_llm": False,
    }


def _prepare_inputs(
    tokenizer,
    model,
    *,
    messages=None,
    plain_text=None,
    input_template_mode="simple_chat_template",
    enable_thinking=False,
):
    prompt_text = None
    input_mode = "plain_text"
    chat_template_used = False
    system_prompt_preview = ""
    user_prompt_preview = ""

    if messages:
        for item in messages:
            if item.get("role") == "system" and not system_prompt_preview:
                system_prompt_preview = item.get("content", "")
            if item.get("role") == "user" and not user_prompt_preview:
                user_prompt_preview = item.get("content", "")

    if messages is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            chat_template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if enable_thinking is not None:
                chat_template_kwargs["enable_thinking"] = enable_thinking
            try:
                prompt_text = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
            except TypeError:
                chat_template_kwargs.pop("enable_thinking", None)
                prompt_text = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
            input_mode = input_template_mode
            chat_template_used = bool(prompt_text)
        except Exception:
            prompt_text = None

    if not prompt_text and plain_text is not None:
        prompt_text = plain_text
        input_mode = "plain_text"
        user_prompt_preview = plain_text
        if not system_prompt_preview:
            system_prompt_preview = SYSTEM_PROMPT

    if not prompt_text and messages is not None:
        prompt_text = "\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages)
        prompt_text += "\n\nASSISTANT:\n"
        input_mode = "prompt_builder_fallback"

    # Preserve the template verbatim: Qwen's empty <think> block is the
    # no-thinking prefix, and removing it changes the model's input tokens.
    # Rendered chat templates already contain their BOS/EOS and role tokens.
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=not chat_template_used)
    device = _get_model_device(model)
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}

    tokenized_input_length = int(encoded["input_ids"].shape[1])
    try:
        decoded_prompt_preview = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
    except Exception:
        decoded_prompt_preview = prompt_text

    return PreparedInput(
        encoded=encoded,
        prompt_text=prompt_text,
        tokenized_input_length=tokenized_input_length,
        decoded_prompt_preview=decoded_prompt_preview,
        input_mode=input_mode,
        chat_template_used=chat_template_used,
        system_prompt_preview=system_prompt_preview,
        user_prompt_preview=user_prompt_preview,
    )


def _resolve_eos_token_id(tokenizer):
    try:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end = None
    if im_end is not None and im_end != getattr(tokenizer, "unk_token_id", None):
        return im_end
    return tokenizer.eos_token_id


def _build_generation_kwargs(tokenizer, generation_defaults):
    generation_kwargs = {
        "max_new_tokens": int(generation_defaults.get("max_new_tokens", 128)),
        "do_sample": bool(generation_defaults.get("do_sample", True)),
        "top_p": float(generation_defaults.get("top_p", 0.9)),
        "repetition_penalty": float(generation_defaults.get("repetition_penalty", 1.0)),
        "use_cache": bool(generation_defaults.get("use_cache", True)),
        "pad_token_id": (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ),
        "eos_token_id": _resolve_eos_token_id(tokenizer),
    }

    top_k = generation_defaults.get("top_k", None)
    if top_k is not None and top_k != "" and top_k is not False:
        generation_kwargs["top_k"] = int(top_k)

    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = float(generation_defaults.get("temperature", 0.7))

    cache_implementation = generation_defaults.get("cache_implementation", "dynamic")
    if generation_kwargs["use_cache"] and cache_implementation and str(cache_implementation).lower() != "dynamic":
        generation_kwargs["cache_implementation"] = str(cache_implementation)

    return generation_kwargs


def _is_llama_cpp_bundle(bundle):
    load_debug = getattr(bundle, "load_debug", {}) or {}
    return str(load_debug.get("backend", "")).strip().lower() == "llama_cpp"


def _build_llama_cpp_generation_kwargs(generation_defaults, llm_seed):
    generation_kwargs = {
        "max_tokens": int(generation_defaults.get("max_new_tokens", 128)),
        "top_p": float(generation_defaults.get("top_p", 0.9)),
        "repeat_penalty": float(generation_defaults.get("repetition_penalty", 1.0)),
        "seed": int(llm_seed),
    }

    if bool(generation_defaults.get("do_sample", True)):
        generation_kwargs["temperature"] = float(generation_defaults.get("temperature", 0.7))
    else:
        generation_kwargs["temperature"] = 0.0

    top_k = generation_defaults.get("top_k", None)
    if top_k is not None and top_k != "" and top_k is not False:
        generation_kwargs["top_k"] = int(top_k)

    return generation_kwargs


def _apply_llama_cpp_thinking_control(messages, *, enable_thinking, thinking_suppression_resolved):
    if enable_thinking or thinking_suppression_resolved != "no_think":
        return messages, False

    updated = []
    applied = False
    for item in messages or []:
        copied = dict(item)
        if copied.get("role") == "user" and not applied:
            content = str(copied.get("content", ""))
            if "/no_think" not in content:
                copied["content"] = content.rstrip() + "\n\n/no_think"
                applied = True
        updated.append(copied)
    return updated, applied


def _extract_llama_cpp_text(response):
    try:
        choices = response.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if content is not None:
                return str(content)
        if choice.get("text") is not None:
            return str(choice.get("text"))
    except Exception:
        pass
    return str(response or "")


def _build_llama_cpp_raw_prompt(messages, *, force_empty_think_block=False):
    chunks = []
    for item in messages or []:
        role = str(item.get("role", "")).strip() or "user"
        content = str(item.get("content", ""))
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    chunks.append("<|im_start|>assistant")
    if force_empty_think_block:
        chunks.append("<think>\n\n</think>\n\n")
    return "\n".join(chunks)


def _generate_llama_cpp_completion_once(model, prompt_text, generation_kwargs):
    completion_kwargs = {
        "max_tokens": generation_kwargs.get("max_tokens"),
        "temperature": generation_kwargs.get("temperature", 0.7),
        "top_p": generation_kwargs.get("top_p", 0.9),
        "repeat_penalty": generation_kwargs.get("repeat_penalty", 1.0),
        "seed": generation_kwargs.get("seed"),
        "stop": generation_kwargs.get("stop", ["<|im_end|>"]),
        "echo": False,
    }
    if "seed" not in generation_kwargs:
        completion_kwargs.pop("seed", None)
    if generation_kwargs.get("stream"):
        completion_kwargs["stream"] = True
    if generation_kwargs.get("top_k") is not None:
        completion_kwargs["top_k"] = generation_kwargs.get("top_k")
    return model(prompt_text, **completion_kwargs)


def _generate_llama_cpp_once_unchecked(model, messages, generation_kwargs, raw_prompt_text=None):
    if raw_prompt_text:
        try:
            return _generate_llama_cpp_completion_once(model, raw_prompt_text, generation_kwargs)
        except TypeError as exc:
            if "seed" not in str(exc):
                raise GenerationError(str(exc), is_oom=False) from exc
            fallback_kwargs = dict(generation_kwargs)
            fallback_kwargs.pop("seed", None)
            try:
                return _generate_llama_cpp_completion_once(model, raw_prompt_text, fallback_kwargs)
            except Exception as fallback_exc:
                raise GenerationError(str(fallback_exc), is_oom=_is_oom_error(fallback_exc, None)) from fallback_exc
        except Exception as exc:
            message = str(exc)
            raise GenerationError(message, is_oom=_is_oom_error(exc, None)) from exc

    try:
        return model.create_chat_completion(
            messages=messages,
            **generation_kwargs,
        )
    except TypeError as exc:
        if "seed" not in str(exc):
            raise GenerationError(str(exc), is_oom=False) from exc
        fallback_kwargs = dict(generation_kwargs)
        fallback_kwargs.pop("seed", None)
        try:
            return model.create_chat_completion(
                messages=messages,
                **fallback_kwargs,
            )
        except Exception as fallback_exc:
            raise GenerationError(str(fallback_exc), is_oom=_is_oom_error(fallback_exc, None)) from fallback_exc
    except Exception as exc:
        message = str(exc)
        raise GenerationError(message, is_oom=_is_oom_error(exc, None)) from exc


def _log_llama_cpp_generation_context(
    logger,
    *,
    bundle,
    gen_prompt,
    original_prompt,
    negative_prompt,
    messages,
    generation_kwargs,
    input_template_mode,
    original_prompt_injected_to_llm,
    llm_seed,
    seed_mode,
    enable_thinking,
    no_think_suffix_applied=False,
):
    if not logger:
        return

    load_debug = getattr(bundle, "load_debug", {}) or {}
    system_prompt_preview = ""
    user_prompt_preview = ""
    for item in messages or []:
        if item.get("role") == "system" and not system_prompt_preview:
            system_prompt_preview = item.get("content", "")
        if item.get("role") == "user" and not user_prompt_preview:
            user_prompt_preview = item.get("content", "")

    _safe_log(logger, f"gen_prompt_raw={gen_prompt}")
    _safe_log(logger, f"original_prompt_raw={original_prompt}")
    _safe_log(logger, f"negative_prompt_raw={negative_prompt}")
    _safe_log(logger, f"seed_mode={seed_mode}")
    _safe_log(logger, f"llm_seed={llm_seed}")
    _safe_log(logger, f"enable_thinking={enable_thinking}")
    _safe_log(logger, f"llm_input_no_think_suffix_applied={no_think_suffix_applied}")
    _safe_log(logger, f"input_template_mode={input_template_mode}")
    _safe_log(logger, f"original_prompt_injected_to_llm={original_prompt_injected_to_llm}")
    _safe_log(logger, "llm_input_mode=llama_cpp_chat_completion")
    _safe_log(logger, f"llm_input_chat_template_source={load_debug.get('chat_template_source')}")
    _safe_log(logger, f"system_prompt_preview={_truncate(system_prompt_preview, 240)}")
    _safe_log(logger, f"user_prompt_preview={_truncate(user_prompt_preview, 240)}")
    _safe_log(logger, "llm_input_messages_full_start")
    _safe_log(logger, str(messages))
    _safe_log(logger, "llm_input_messages_full_end")
    _safe_log(
        logger,
        "generation_params "
        f"max_tokens={generation_kwargs.get('max_tokens')} "
        f"temperature={generation_kwargs.get('temperature')} "
        f"top_p={generation_kwargs.get('top_p')} "
        f"top_k={generation_kwargs.get('top_k')} "
        f"repeat_penalty={generation_kwargs.get('repeat_penalty')} "
        f"seed={generation_kwargs.get('seed')}"
    )


def _generate_prompt_candidates_llama_cpp(
    bundle,
    gen_prompt,
    original_prompt,
    negative_prompt,
    generation_defaults,
    interrupt_checker=None,
    logger=None,
    num_candidates: int = 1,
):
    interrupted_seen = False
    original_checker = interrupt_checker

    def check_interrupt():
        nonlocal interrupted_seen
        interrupted_seen = interrupted_seen or bool(original_checker and original_checker())
        return interrupted_seen

    interrupt_checker = check_interrupt
    input_template_mode = generation_defaults.get("input_template_mode", "simple_chat_template")
    payload = _build_input_payload(
        input_template_mode,
        gen_prompt,
        original_prompt,
        negative_prompt,
    )
    messages = payload["messages"]
    llm_seed, seed_mode = _resolve_llm_seed(generation_defaults)
    enable_thinking = generation_defaults.get("enable_thinking", False)
    load_debug = getattr(bundle, "load_debug", {}) or {}
    thinking_suppression_resolved = str(
        load_debug.get("thinking_suppression_resolved", "none") or "none"
    ).strip().lower()
    gguf_prompt_mode = getattr(getattr(bundle, "spec", None), "gguf_prompt_mode",
                               load_debug.get("gguf_prompt_mode", "legacy"))
    raw_prompt_text = None
    raw_prompt_tokens = None
    no_think_suffix_applied = False
    generation_kwargs = _build_llama_cpp_generation_kwargs(generation_defaults, llm_seed)
    if gguf_prompt_mode == "embedded_template":
        raw_prompt_text, raw_prompt_tokens, template_stop = _render_llama_cpp_embedded_prompt(
            bundle.model, messages, enable_thinking=enable_thinking,
        )
        generation_kwargs["stop"] = template_stop
    else:
        messages, no_think_suffix_applied = _apply_llama_cpp_thinking_control(
            messages,
            enable_thinking=enable_thinking,
            thinking_suppression_resolved=thinking_suppression_resolved,
        )
        if no_think_suffix_applied:
            raw_prompt_text = _build_llama_cpp_raw_prompt(
                messages,
                force_empty_think_block=True,
            )
    effective_num_candidates = max(1, int(num_candidates))

    _log_llama_cpp_generation_context(
        logger,
        bundle=bundle,
        gen_prompt=gen_prompt,
        original_prompt=original_prompt,
        negative_prompt=negative_prompt,
        messages=messages,
        generation_kwargs=generation_kwargs,
        input_template_mode=payload["input_template_mode"],
        original_prompt_injected_to_llm=payload["original_prompt_injected_to_llm"],
        llm_seed=llm_seed,
        seed_mode=seed_mode,
        enable_thinking=enable_thinking,
        no_think_suffix_applied=no_think_suffix_applied,
    )

    if interrupt_checker and interrupt_checker():
        return GenerationResult(
            llm_called=False,
            llm_seed=llm_seed,
            seed_mode=seed_mode,
            input_debug={
                "messages": messages,
                "input_mode": "llama_cpp_embedded_template" if gguf_prompt_mode == "embedded_template" else "llama_cpp_chat_completion",
                "gguf_prompt_mode": gguf_prompt_mode,
                "raw_prompt_text": raw_prompt_text,
                "tokenized_input_length": len(raw_prompt_tokens) if raw_prompt_tokens is not None else None,
                "input_template_mode": payload["input_template_mode"],
                "original_prompt_injected_to_llm": payload["original_prompt_injected_to_llm"],
                "generation_kwargs": generation_kwargs,
                "llm_seed": llm_seed,
                "seed_mode": seed_mode,
                "enable_thinking": enable_thinking,
                "thinking_suppression_resolved": thinking_suppression_resolved,
                "no_think_suffix_applied": no_think_suffix_applied,
            },
            interrupted=True,
        )

    started_at = time.perf_counter()
    candidate_texts = []
    candidate_debugs = []
    chosen_index = -1
    chosen_cleaned = ""
    chosen_reject_reason = ""
    try:
        for idx in range(effective_num_candidates):
            if interrupt_checker and interrupt_checker():
                return GenerationResult(
                    llm_called=bool(candidate_texts),
                    interrupted=True,
                    llm_seed=llm_seed,
                    seed_mode=seed_mode,
                )
            per_candidate_kwargs = dict(generation_kwargs)
            per_candidate_kwargs["seed"] = int(llm_seed) + idx
            response = _generate_llama_cpp_once(
                bundle.model,
                messages,
                per_candidate_kwargs,
                raw_prompt_text=raw_prompt_text,
                interrupt_checker=interrupt_checker,
                raw_prompt_tokens=raw_prompt_tokens,
            )
            if interrupt_checker():
                return GenerationResult(llm_called=True, interrupted=True,
                                        llm_seed=llm_seed, seed_mode=seed_mode)
            raw_output = _extract_llama_cpp_text(response)
            normalized_output = normalize_model_output(raw_output)
            cleaned_output = clean_generated_positive(normalized_output)
            usable, reject_reason = is_generated_prompt_strong_enough(cleaned_output)
            accepted_for_this = usable and chosen_index < 0
            candidate_texts.append(cleaned_output)
            candidate_debugs.append(
                _make_candidate_debug(
                    index=idx,
                    raw_output=raw_output,
                    normalized_output=normalized_output,
                    cleaned_output=cleaned_output,
                    accepted=accepted_for_this,
                    reject_reason=reject_reason,
                )
            )
            if accepted_for_this:
                chosen_index = idx
                chosen_cleaned = cleaned_output
                chosen_reject_reason = ""
        if chosen_index < 0 and candidate_debugs:
            chosen_index = 0
            chosen_cleaned = candidate_texts[0]
            chosen_reject_reason = candidate_debugs[0].get("reject_reason", "")
    except Exception:
        raise

    generate_seconds = time.perf_counter() - started_at
    _safe_log(logger, f"llm_generate_seconds={generate_seconds:.3f} num_candidates={effective_num_candidates}")

    chosen_debug = candidate_debugs[chosen_index] if chosen_index >= 0 else {}
    usable = chosen_debug.get("accepted", False) if chosen_debug else False
    fallback_used = not usable
    fallback_reason = chosen_reject_reason if fallback_used else ""
    selected_positive = chosen_cleaned if usable else build_manual_positive(gen_prompt, "")

    return GenerationResult(
        llm_called=True,
        candidates=[t for t in candidate_texts if t],
        candidate_debug=candidate_debugs,
        chosen_candidate=chosen_cleaned,
        chosen_candidate_debug=chosen_debug,
        selected_positive=selected_positive,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        llm_seed=llm_seed,
        seed_mode=seed_mode,
        input_debug={
            "messages": messages,
            "input_mode": "llama_cpp_embedded_template" if gguf_prompt_mode == "embedded_template" else "llama_cpp_chat_completion",
                "gguf_prompt_mode": gguf_prompt_mode,
                "raw_prompt_text": raw_prompt_text,
                "tokenized_input_length": len(raw_prompt_tokens) if raw_prompt_tokens is not None else None,
            "input_template_mode": payload["input_template_mode"],
            "original_prompt_injected_to_llm": payload["original_prompt_injected_to_llm"],
            "generation_kwargs": generation_kwargs,
            "llm_seed": llm_seed,
            "seed_mode": seed_mode,
            "enable_thinking": enable_thinking,
            "thinking_suppression_resolved": thinking_suppression_resolved,
            "no_think_suffix_applied": no_think_suffix_applied,
        },
        interrupted=False,
        generate_seconds=generate_seconds,
    )


def _generate_once(model, inputs, generation_kwargs, torch):
    with torch.no_grad():
        return model.generate(**inputs, **generation_kwargs)


def _resolve_llm_seed(generation_defaults):
    seed_mode = str(generation_defaults.get("seed_mode", "random")).strip().lower()
    if seed_mode == "fixed":
        return int(generation_defaults.get("llm_seed", 42)), "fixed"
    return secrets.randbelow(2**31 - 1) + 1, "random"


def _apply_llm_seed(torch, llm_seed):
    torch.manual_seed(int(llm_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(llm_seed))


def _clear_generation_state(torch, aggressive: bool = True):
    if not aggressive:
        return
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _make_candidate_debug(index, raw_output, normalized_output, cleaned_output, accepted, reject_reason):
    return {
        "index": index,
        "raw_model_output": raw_output,
        "normalized_output": normalized_output,
        "cleaned_output": cleaned_output,
        "accepted": accepted,
        "reject_reason": reject_reason,
    }


def _log_generation_context(
    logger,
    *,
    bundle,
    gen_prompt,
    original_prompt,
    negative_prompt,
    prepared_input,
    generation_kwargs,
    input_template_mode,
    original_prompt_injected_to_llm,
    llm_seed,
    seed_mode,
    enable_thinking,
):
    if not logger:
        return

    load_debug = getattr(bundle, "load_debug", {}) or {}
    _safe_log(logger, f"gen_prompt_raw={gen_prompt}")
    _safe_log(logger, f"original_prompt_raw={original_prompt}")
    _safe_log(logger, f"negative_prompt_raw={negative_prompt}")
    _safe_log(logger, f"seed_mode={seed_mode}")
    _safe_log(logger, f"llm_seed={llm_seed}")
    _safe_log(logger, f"enable_thinking={enable_thinking}")
    _safe_log(logger, f"input_template_mode={input_template_mode}")
    _safe_log(logger, f"original_prompt_injected_to_llm={original_prompt_injected_to_llm}")
    _safe_log(logger, f"llm_input_mode={prepared_input.input_mode}")
    _safe_log(logger, f"llm_input_chat_template_used={prepared_input.chat_template_used}")
    _safe_log(logger, f"llm_input_chat_template_source={load_debug.get('chat_template_source')}")
    _safe_log(logger, f"llm_input_empty_think_block_stripped={prepared_input.empty_think_block_stripped}")
    _safe_log(logger, f"system_prompt_preview={_truncate(prepared_input.system_prompt_preview, 240)}")
    _safe_log(logger, f"user_prompt_preview={_truncate(prepared_input.user_prompt_preview, 240)}")
    _safe_log(logger, "llm_input_text_full_start")
    _safe_log(logger, prepared_input.prompt_text)
    _safe_log(logger, "llm_input_text_full_end")
    _safe_log(logger, f"tokenized_input_length={prepared_input.tokenized_input_length}")
    _safe_log(logger, f"decoded_prompt_preview={_truncate(prepared_input.decoded_prompt_preview, 600)}")
    _safe_log(
        logger,
        "generation_params "
        f"max_new_tokens={generation_kwargs.get('max_new_tokens')} "
        f"temperature={generation_kwargs.get('temperature')} "
        f"top_p={generation_kwargs.get('top_p')} "
        f"top_k={generation_kwargs.get('top_k')} "
        f"do_sample={generation_kwargs.get('do_sample')} "
        f"repetition_penalty={generation_kwargs.get('repetition_penalty')} "
        f"cache_implementation={generation_kwargs.get('cache_implementation', 'dynamic(default)')} "
        f"use_cache={generation_kwargs.get('use_cache')}"
    )


def debug_compare_prompt_variants(
    *,
    bundle,
    gen_prompt,
    original_prompt,
    negative_prompt,
    generation_defaults,
    logger=None,
    interrupt_checker=None,
):
    torch = _import_torch()
    tokenizer = bundle.tokenizer
    model = bundle.model
    generation_kwargs = _build_generation_kwargs(tokenizer, generation_defaults)
    comparison_kwargs = dict(generation_kwargs)
    comparison_kwargs["max_new_tokens"] = min(int(comparison_kwargs.get("max_new_tokens", 128)), 64)
    llm_seed, seed_mode = _resolve_llm_seed(generation_defaults)
    enable_thinking = generation_defaults.get("enable_thinking", False)

    variants = [
        {
            "name": "forge_prompt_builder",
            "payload": _build_input_payload("forge_prompt_builder", gen_prompt, original_prompt, negative_prompt),
        },
        {
            "name": "plain_text_only",
            "payload": {
                "messages": None,
                "plain_text": build_plain_text_input(gen_prompt, ""),
                "input_template_mode": "plain_text_only",
            },
        },
        {
            "name": "simple_chat_template",
            "payload": _build_input_payload("simple_chat_template", gen_prompt, original_prompt, negative_prompt),
        },
    ]

    for variant in variants:
        if interrupt_checker is not None and interrupt_checker():
            return True
        criterion = None
        if interrupt_checker is not None:
            stopping_criteria, criterion = _interrupt_stopping_criteria(interrupt_checker, torch)
            comparison_kwargs["stopping_criteria"] = stopping_criteria
        try:
            prepared = _prepare_inputs(
                tokenizer,
                model,
                messages=variant["payload"]["messages"],
                plain_text=variant["payload"]["plain_text"],
                input_template_mode=variant["payload"]["input_template_mode"],
                enable_thinking=enable_thinking,
            )
            with _isolated_llm_rng(torch, llm_seed):
                outputs = _generate_once(model, prepared.encoded, comparison_kwargs, torch)
            if criterion is not None and criterion.was_interrupted:
                return True
            generated_tokens = outputs[0][prepared.encoded["input_ids"].shape[1]:]
            raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            normalized_output = normalize_model_output(raw_output)
            cleaned_output = clean_generated_positive(normalized_output)
            _safe_log(
                logger,
                f"debug_variant[{variant['name']}] "
                f"input_mode={prepared.input_mode} "
                f"chat_template_used={prepared.chat_template_used} "
                f"tokenized_input_length={prepared.tokenized_input_length} "
                f"seed_mode={seed_mode} "
                f"llm_seed={llm_seed}"
            )
            _safe_log(logger, f"debug_variant[{variant['name']}] system_prompt={_truncate(prepared.system_prompt_preview, 240)}")
            _safe_log(logger, f"debug_variant[{variant['name']}] user_prompt={_truncate(prepared.user_prompt_preview, 240)}")
            _safe_log(logger, f"debug_variant[{variant['name']}] input_text={_truncate(prepared.prompt_text, 600)}")
            _safe_log(logger, f"debug_variant[{variant['name']}] raw_output={_truncate(raw_output, 300)}")
            _safe_log(logger, f"debug_variant[{variant['name']}] normalized_output={_truncate(normalized_output, 300)}")
            _safe_log(logger, f"debug_variant[{variant['name']}] cleaned_output={_truncate(cleaned_output, 300)}")
        except Exception as exc:
            if _is_oom_error(exc, torch):
                raise
            _safe_log(logger, f"debug_variant[{variant['name']}] failed={exc}")
        finally:
            prepared = outputs = generated_tokens = None
        _clear_generation_state(torch)
    return False


def generate_prompt_candidates(
    bundle,
    gen_prompt,
    original_prompt,
    negative_prompt,
    generation_defaults,
    interrupt_checker=None,
    logger=None,
    aggressive_cleanup: bool = True,
    num_candidates: int = 1,
):
    if _is_llama_cpp_bundle(bundle):
        try:
            return _generate_prompt_candidates_llama_cpp(
                bundle=bundle,
                gen_prompt=gen_prompt,
                original_prompt=original_prompt,
                negative_prompt=negative_prompt,
                generation_defaults=generation_defaults,
                interrupt_checker=interrupt_checker,
                logger=logger,
                num_candidates=num_candidates,
            )
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(str(exc), is_oom=_is_oom_error(exc, None)) from exc

    torch = _import_torch()
    try:
        result = _generate_transformers_candidates(
            torch, bundle, gen_prompt, original_prompt, negative_prompt,
            generation_defaults, interrupt_checker, logger, num_candidates,
        )
    except GenerationError:
        raise
    except Exception as exc:
        raise GenerationError(str(exc), is_oom=_is_oom_error(exc, torch)) from exc
    # The inner frame no longer owns inputs/outputs. OOM cleanup is deferred to unload.
    _clear_generation_state(torch, aggressive=aggressive_cleanup)
    return result


def _generate_transformers_candidates(
    torch, bundle, gen_prompt, original_prompt, negative_prompt,
    generation_defaults, interrupt_checker, logger, num_candidates,
):
    input_template_mode = generation_defaults.get("input_template_mode", "simple_chat_template")
    payload = _build_input_payload(
        input_template_mode,
        gen_prompt,
        original_prompt,
        negative_prompt,
    )
    model = bundle.model
    tokenizer = bundle.tokenizer
    llm_seed, seed_mode = _resolve_llm_seed(generation_defaults)
    enable_thinking = generation_defaults.get("enable_thinking", False)
    prepared_input = _prepare_inputs(
        tokenizer,
        model,
        messages=payload["messages"],
        plain_text=payload["plain_text"],
        input_template_mode=payload["input_template_mode"],
        enable_thinking=enable_thinking,
    )
    generation_kwargs = _build_generation_kwargs(tokenizer, generation_defaults)
    effective_num_candidates = max(1, int(num_candidates))
    if effective_num_candidates > 1:
        generation_kwargs["num_return_sequences"] = effective_num_candidates

    _log_generation_context(
        logger,
        bundle=bundle,
        gen_prompt=gen_prompt,
        original_prompt=original_prompt,
        negative_prompt=negative_prompt,
        prepared_input=prepared_input,
        generation_kwargs=generation_kwargs,
        input_template_mode=payload["input_template_mode"],
        original_prompt_injected_to_llm=payload["original_prompt_injected_to_llm"],
        llm_seed=llm_seed,
        seed_mode=seed_mode,
        enable_thinking=enable_thinking,
    )

    debug_interrupted = False
    if generation_defaults.get("debug_compare_input_variants", False):
        debug_interrupted = debug_compare_prompt_variants(
            bundle=bundle,
            gen_prompt=gen_prompt,
            original_prompt=original_prompt,
            negative_prompt=negative_prompt,
            generation_defaults=generation_defaults,
            logger=logger,
            interrupt_checker=interrupt_checker,
        )

    if debug_interrupted or (interrupt_checker and interrupt_checker()):
        return GenerationResult(
            llm_called=False,
            llm_seed=llm_seed,
            seed_mode=seed_mode,
            input_debug={
                "llm_input_text_full": prepared_input.prompt_text,
                "tokenized_input_length": prepared_input.tokenized_input_length,
                "decoded_prompt_preview": prepared_input.decoded_prompt_preview,
                "input_mode": prepared_input.input_mode,
                "input_template_mode": payload["input_template_mode"],
                "original_prompt_injected_to_llm": payload["original_prompt_injected_to_llm"],
                "chat_template_used": prepared_input.chat_template_used,
                "system_prompt_preview": prepared_input.system_prompt_preview,
                "user_prompt_preview": prepared_input.user_prompt_preview,
                "generation_kwargs": generation_kwargs,
                "llm_seed": llm_seed,
                "seed_mode": seed_mode,
                "enable_thinking": enable_thinking,
            },
            interrupted=True,
        )

    criterion = None
    if interrupt_checker is not None:
        stopping_criteria, criterion = _interrupt_stopping_criteria(interrupt_checker, torch)
        generation_kwargs["stopping_criteria"] = stopping_criteria
    # Greedy decoding cannot request multiple sequences without beams.
    if effective_num_candidates > 1 and not generation_kwargs.get("do_sample", True):
        generation_kwargs["num_beams"] = effective_num_candidates
    started_at = time.perf_counter()
    input_length = prepared_input.encoded["input_ids"].shape[1]
    candidate_texts = []
    candidate_debugs = []
    chosen_index = -1
    chosen_cleaned = ""
    chosen_reject_reason = ""
    try:
        with _isolated_llm_rng(torch, llm_seed):
            output_ids = _generate_once(model, prepared_input.encoded, generation_kwargs, torch)
        if ((criterion is not None and criterion.was_interrupted)
                or (interrupt_checker is not None and interrupt_checker())):
            return GenerationResult(llm_called=True, interrupted=True, llm_seed=llm_seed,
                                    seed_mode=seed_mode, generate_seconds=time.perf_counter() - started_at)
        batch_size = output_ids.shape[0]
        for idx in range(batch_size):
            generated_tokens = output_ids[idx][input_length:]
            raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            normalized_output = normalize_model_output(raw_output)
            cleaned_output = clean_generated_positive(normalized_output)
            usable, reject_reason = is_generated_prompt_strong_enough(cleaned_output)
            accepted_for_this = usable and chosen_index < 0
            candidate_texts.append(cleaned_output)
            candidate_debugs.append(
                _make_candidate_debug(
                    index=idx,
                    raw_output=raw_output,
                    normalized_output=normalized_output,
                    cleaned_output=cleaned_output,
                    accepted=accepted_for_this,
                    reject_reason=reject_reason,
                )
            )
            if accepted_for_this:
                chosen_index = idx
                chosen_cleaned = cleaned_output
                chosen_reject_reason = ""
        if chosen_index < 0 and candidate_debugs:
            chosen_index = 0
            chosen_cleaned = candidate_texts[0]
            chosen_reject_reason = candidate_debugs[0].get("reject_reason", "")
    except Exception:
        raise
    generate_seconds = time.perf_counter() - started_at
    _safe_log(logger, f"llm_generate_seconds={generate_seconds:.3f} num_candidates={effective_num_candidates}")

    chosen_debug = candidate_debugs[chosen_index] if chosen_index >= 0 else {}
    usable = chosen_debug.get("accepted", False) if chosen_debug else False
    fallback_used = not usable
    fallback_reason = chosen_reject_reason if fallback_used else ""
    selected_positive = chosen_cleaned if usable else build_manual_positive(gen_prompt, "")

    return GenerationResult(
        llm_called=True,
        candidates=[t for t in candidate_texts if t],
        candidate_debug=candidate_debugs,
        chosen_candidate=chosen_cleaned,
        chosen_candidate_debug=chosen_debug,
        selected_positive=selected_positive,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        llm_seed=llm_seed,
        seed_mode=seed_mode,
        input_debug={
            "llm_input_text_full": prepared_input.prompt_text,
            "tokenized_input_length": prepared_input.tokenized_input_length,
            "decoded_prompt_preview": prepared_input.decoded_prompt_preview,
            "input_mode": prepared_input.input_mode,
            "input_template_mode": payload["input_template_mode"],
            "original_prompt_injected_to_llm": payload["original_prompt_injected_to_llm"],
            "chat_template_used": prepared_input.chat_template_used,
            "system_prompt_preview": prepared_input.system_prompt_preview,
            "user_prompt_preview": prepared_input.user_prompt_preview,
            "generation_kwargs": generation_kwargs,
            "llm_seed": llm_seed,
            "seed_mode": seed_mode,
            "enable_thinking": enable_thinking,
            "empty_think_block_stripped": prepared_input.empty_think_block_stripped,
        },
        interrupted=False,
        generate_seconds=generate_seconds,
    )


@contextmanager
def _isolated_llm_rng(torch, llm_seed):
    # The extension shares a process with Forge. Restore both CPU and GPU RNG
    # state even on failure so LLM sampling cannot change image generation.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        _apply_llm_seed(torch, llm_seed)
        yield


def _interrupt_stopping_criteria(interrupt_checker, torch):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class InterruptStoppingCriteria(StoppingCriteria):
        was_interrupted = False

        def __call__(self, input_ids, scores, **kwargs):
            self.was_interrupted = self.was_interrupted or bool(interrupt_checker())
            return torch.full(
                (input_ids.shape[0],), self.was_interrupted,
                dtype=torch.bool, device=input_ids.device,
            )

    criterion = InterruptStoppingCriteria()
    return StoppingCriteriaList([criterion]), criterion


def _is_oom_error(exc, torch):
    return (
        (isinstance(exc, GenerationError) and exc.is_oom)
        or isinstance(exc, (MemoryError, getattr(torch, "OutOfMemoryError", MemoryError)))
        or "out of memory" in str(exc).lower()
    )


def _generate_llama_cpp_once(model, messages, generation_kwargs, raw_prompt_text=None,
                             interrupt_checker=None, raw_prompt_tokens=None):
    # A token list bypasses Llama's string tokenization and a second BOS insertion.
    completion_prompt = raw_prompt_tokens if raw_prompt_tokens is not None else raw_prompt_text
    kwargs = dict(generation_kwargs)
    if interrupt_checker is None:
        return _generate_llama_cpp_once_unchecked(model, messages, kwargs, completion_prompt)
    kwargs["stream"] = True
    stream = _generate_llama_cpp_once_unchecked(model, messages, kwargs, completion_prompt)
    parts = []
    try:
        for chunk in stream:
            if interrupt_checker():
                break
            choices = chunk.get("choices", [])
            if choices:
                choice = choices[0]
                parts.append(choice.get("text") or choice.get("delta", {}).get("content") or "")
    except GenerationError:
        raise
    except Exception as exc:
        raise GenerationError(str(exc), is_oom=_is_oom_error(exc, None)) from exc
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    return {"choices": [{"text": "".join(parts)}]}


def _render_llama_cpp_embedded_prompt(model, messages, *, enable_thinking=False):
    """Use GGUF metadata verbatim and retain its no-thinking/special-token policy."""
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    metadata = getattr(model, "metadata", {}) or {}
    template = next((metadata[key] for key in (
        "tokenizer.chat_template", "tokenizer.ggml.chat_template", "chat_template",
    ) if isinstance(metadata.get(key), str) and metadata[key].strip()), None)
    if template is None:
        raise GenerationError("embedded_template requires a GGUF tokenizer.chat_template; no legacy fallback was applied.")

    def special_token(token_id):
        if token_id is None or token_id < 0:
            return ""
        return model.detokenize([token_id], special=True).decode("utf-8")

    formatter = Jinja2ChatFormatter(
        template=template,
        eos_token=special_token(model.token_eos()),
        bos_token=special_token(model.token_bos()),
        add_generation_prompt=True,
    )
    rendered = formatter(messages=messages, enable_thinking=bool(enable_thinking))
    if not rendered.prompt:
        raise GenerationError("GGUF embedded template rendered an empty prompt.")
    tokens = model.tokenize(rendered.prompt.encode("utf-8"), add_bos=False, special=True)
    if not tokens:
        raise GenerationError("GGUF embedded template produced no input tokens.")
    return rendered.prompt, tokens, rendered.stop
