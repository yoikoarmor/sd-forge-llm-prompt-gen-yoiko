import re


LEGACY_SYSTEM_PROMPT = (
    "You rewrite short image-generation requests into strong positive prompts for Stable Diffusion. "
    "Return exactly one single-line positive prompt. "
    "Output only comma-separated tags or a short prompt phrase. "
    "Do not include explanations, headings, bullets, markdown, or labels like 'Prompt:'. "
    "Do not copy negative prompt items into the positive prompt."
)

SYSTEM_PROMPT = (
    "You are a prompt writer for image generation.\n"
    "Rewrite the user's input into a high-quality positive prompt for image generation.\n"
    "Preserve the core subjects, attributes, and intent from the user's input.\n"
    "Do not remove important elements unless they are clearly invalid noise.\n"
    "You may add a moderate amount of visual, situational, or atmospheric detail to make the prompt more useful for image generation.\n"
    "Keep additions subtle and grounded.\n"
    "Avoid overly vivid color language, excessive dramatic emphasis, or flashy embellishment unless the user's input clearly asks for it.\n"
    "Prefer natural, restrained visual detail over exaggerated stylization.\n"
    "Do not radically change the theme.\n"
    "Do not explain anything.\n"
    "Do not output headers such as \"Prompt:\".\n"
    "Return only the positive prompt text in natural comma-separated prompt style.\n"
    "Do not include negative prompt terms in the positive prompt."
)


def _to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


PREFIXES = [
    "prompt:",
    "positive prompt:",
    "generated prompt:",
    "generated_positive:",
    "answer:",
    "assistant:",
    "response:",
]


def build_messages(gen_prompt, original_prompt, negative_prompt):
    gen_prompt = _to_text(gen_prompt).strip()
    negative_prompt = _to_text(negative_prompt).strip()

    sections = []
    if gen_prompt:
        sections.append(f"User input:\n{gen_prompt}")
    if negative_prompt:
        sections.append(f"Negative prompt reference (do not include these terms):\n{negative_prompt}")
    sections.append(
        "Rewrite this into a better positive prompt for image generation. "
        "Preserve the original idea and main traits, and add only a moderate amount of useful visual detail."
    )

    user_content = "\n\n".join(sections) if sections else "Write a high-quality positive prompt."

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_simple_chat_messages(gen_prompt, original_prompt, negative_prompt=""):
    gen_prompt = _to_text(gen_prompt).strip()
    negative_prompt = _to_text(negative_prompt).strip()

    sections = []
    if gen_prompt:
        sections.append(f"User input:\n{gen_prompt}")
    if negative_prompt:
        sections.append(f"Negative prompt reference (do not include these terms):\n{negative_prompt}")
    sections.append(
        "Rewrite this into a better positive prompt for image generation. "
        "Preserve the original idea and main traits, and add only a moderate amount of useful visual detail."
    )
    user_prompt = "\n\n".join(section for section in sections if section)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_plain_text_input(gen_prompt, original_prompt):
    gen_prompt = _to_text(gen_prompt).strip()
    return gen_prompt


def normalize_model_output(text):
    text = _to_text(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    return text


def _strip_known_prefix(text):
    lowered = text.lower()
    for prefix in PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _strip_wrapping_quotes(text):
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "`"}:
        return text[1:-1].strip()
    return text


def clean_generated_positive(text):
    text = normalize_model_output(text)
    if not text:
        return ""

    text = _strip_known_prefix(text)
    text = _strip_wrapping_quotes(text.strip(" "))
    joined = text.replace("\n", ", ")
    joined = re.sub(r"\s+", " ", joined)
    joined = re.sub(r"\s*,\s*", ", ", joined).strip(" ,")

    if joined.lower() in {"none", "n/a", "null"}:
        return ""

    return joined


def dedupe_prompt_terms(text):
    cleaned = clean_generated_positive(text)
    if not cleaned:
        return "", False

    seen = set()
    deduped_terms = []
    dedupe_applied = False

    for raw_term in cleaned.split(","):
        term = raw_term.strip()
        if not term:
            continue
        if term in seen:
            dedupe_applied = True
            continue
        seen.add(term)
        deduped_terms.append(term)

    deduped_text = ", ".join(deduped_terms)
    if deduped_text != cleaned:
        dedupe_applied = True
    return deduped_text, dedupe_applied


def sanitize_generated_positive(text):
    return clean_generated_positive(text)


def is_generated_prompt_strong_enough(text):
    cleaned = clean_generated_positive(text)
    if not cleaned:
        return False, "empty"

    prompt_parts = [item.strip() for item in cleaned.split(",") if item.strip()]
    if len(cleaned) < 10:
        return False, "too_short_chars"
    if len(prompt_parts) < 2:
        return False, "too_few_prompt_parts"

    return True, ""


def build_manual_positive(gen_prompt, original_prompt):
    gen_prompt = _to_text(gen_prompt).strip()
    original_prompt = _to_text(original_prompt).strip()

    if not gen_prompt:
        combined = original_prompt
    elif not original_prompt:
        combined = gen_prompt
    else:
        combined = f"{gen_prompt}, {original_prompt}"

    deduped_text, _ = dedupe_prompt_terms(combined)
    return deduped_text


def build_final_positive_details(generated_positive, original_prompt, append_original=True):
    generated_positive = clean_generated_positive(generated_positive)
    original_prompt = clean_generated_positive(original_prompt)

    if generated_positive and original_prompt and append_original:
        before_dedupe = f"{generated_positive}, {original_prompt}"
    elif generated_positive:
        before_dedupe = generated_positive
    else:
        before_dedupe = original_prompt

    after_dedupe, dedupe_applied = dedupe_prompt_terms(before_dedupe)
    return {
        "final_positive_before_dedupe": before_dedupe,
        "final_positive_after_dedupe": after_dedupe,
        "dedupe_applied": dedupe_applied,
        "original_prompt_appended_after_llm": bool(generated_positive and original_prompt and append_original),
    }


def build_final_positive(generated_positive, original_prompt):
    details = build_final_positive_details(generated_positive, original_prompt, append_original=True)
    return details["final_positive_after_dedupe"]


def summarize_text(text, max_chars=80):
    text = clean_generated_positive(text) if text else _to_text(text).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text
