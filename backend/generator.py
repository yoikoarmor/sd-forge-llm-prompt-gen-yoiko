import gc
import re
import secrets
import time
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


def _strip_empty_think_block(prompt_text, *, enable_thinking):
    if enable_thinking or not prompt_text:
        return prompt_text, False

    stripped = re.sub(
        r"(<\|im_start\|>assistant\s*)<think>\s*</think>\s*",
        r"\1",
        prompt_text,
        count=1,
        flags=re.DOTALL,
    )
    return stripped, stripped != prompt_text


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
            chat_template_used = True
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

    prompt_text, empty_think_block_stripped = _strip_empty_think_block(
        prompt_text,
        enable_thinking=enable_thinking,
    )

    encoded = tokenizer(prompt_text, return_tensors="pt")
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
        empty_think_block_stripped=empty_think_block_stripped,
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
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": _resolve_eos_token_id(tokenizer),
    }

    top_k = generation_defaults.get("top_k", None)
    if top_k not in (None, "", False):
        generation_kwargs["top_k"] = int(top_k)

    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = float(generation_defaults.get("temperature", 0.7))

    cache_implementation = generation_defaults.get("cache_implementation", "dynamic")
    if generation_kwargs["use_cache"] and cache_implementation and str(cache_implementation).lower() != "dynamic":
        generation_kwargs["cache_implementation"] = str(cache_implementation)

    return generation_kwargs


def _generate_once(model, inputs, generation_kwargs, torch):
    try:
        with torch.no_grad():
            return model.generate(**inputs, **generation_kwargs)
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            raise GenerationError(message, is_oom=True) from exc
        raise GenerationError(message, is_oom=False) from exc
    except Exception as exc:
        raise GenerationError(str(exc), is_oom=False) from exc


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
):
    torch = _import_torch()
    tokenizer = bundle.tokenizer
    model = bundle.model
    generation_kwargs = _build_generation_kwargs(tokenizer, generation_defaults)
    comparison_kwargs = dict(generation_kwargs)
    comparison_kwargs["max_new_tokens"] = min(int(comparison_kwargs.get("max_new_tokens", 128)), 64)
    llm_seed, seed_mode = _resolve_llm_seed(generation_defaults)
    enable_thinking = generation_defaults.get("enable_thinking", False)
    _apply_llm_seed(torch, llm_seed)

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
        try:
            prepared = _prepare_inputs(
                tokenizer,
                model,
                messages=variant["payload"]["messages"],
                plain_text=variant["payload"]["plain_text"],
                input_template_mode=variant["payload"]["input_template_mode"],
                enable_thinking=enable_thinking,
            )
            outputs = _generate_once(model, prepared.encoded, comparison_kwargs, torch)
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
            _safe_log(logger, f"debug_variant[{variant['name']}] failed={exc}")
        finally:
            _clear_generation_state(torch)


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
    torch = _import_torch()

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
    _apply_llm_seed(torch, llm_seed)

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

    if generation_defaults.get("debug_compare_input_variants", False):
        debug_compare_prompt_variants(
            bundle=bundle,
            gen_prompt=gen_prompt,
            original_prompt=original_prompt,
            negative_prompt=negative_prompt,
            generation_defaults=generation_defaults,
            logger=logger,
        )

    if interrupt_checker and interrupt_checker():
        return GenerationResult(
            llm_called=False,
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

    started_at = time.perf_counter()
    input_length = prepared_input.encoded["input_ids"].shape[1]
    candidate_texts = []
    candidate_debugs = []
    chosen_index = -1
    chosen_cleaned = ""
    chosen_reject_reason = ""
    try:
        output_ids = _generate_once(model, prepared_input.encoded, generation_kwargs, torch)
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
        _clear_generation_state(torch, aggressive=True)
        raise
    else:
        _clear_generation_state(torch, aggressive=aggressive_cleanup)
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
