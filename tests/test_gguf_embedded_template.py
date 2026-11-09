import sys
import types
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from backend import generator, llama_cpp_loader, registry


class EmbeddedTemplateTests(unittest.TestCase):
    def fixture(self, request="a traveler", *, thinking=False):
        prompt = "<custom>" + request + "</custom><assistant>"
        if not thinking:
            prompt += "<think>\n\n</think>\n\n"
        formatter = Mock(return_value=types.SimpleNamespace(prompt=prompt, stop=["<custom-eos>"], added_special=True))
        factory = Mock(return_value=formatter)
        module = types.ModuleType("llama_cpp.llama_chat_format")
        module.Jinja2ChatFormatter = factory
        model = Mock()
        model.metadata = {"tokenizer.chat_template": "actual embedded template"}
        model.token_eos.return_value = 2
        model.token_bos.return_value = -1
        model.detokenize.return_value = b"<custom-eos>"
        model.tokenize.return_value = [11, 12, 13]
        model.return_value = iter([{"choices": [{"text": "a traveler, soft light"}]}])
        bundle = types.SimpleNamespace(
            spec=registry.ModelSpec("opt-in", backend="llama_cpp", gguf_prompt_mode="embedded_template"),
            model=model, load_debug={"backend": "llama_cpp", "thinking_suppression_resolved": "embedded_template"})
        return bundle, module, factory, formatter, prompt

    def test_opt_in_only_for_new_instruct7_preset(self):
        self.assertEqual(registry.get_model_spec("qwen3.5-9b-lora-7-q4km").gguf_prompt_mode, "embedded_template")
        legacy = [registry.get_model_spec(key) for key in registry.get_ui_model_choices()
                  if key != "none" and key != "qwen3.5-9b-lora-7-q4km"]
        self.assertTrue(all(spec.gguf_prompt_mode == "legacy" for spec in legacy))
        spec = registry.ModelSpec("native", backend="llama_cpp")
        self.assertNotEqual(spec.signature(), replace(spec, gguf_prompt_mode="embedded_template").signature())
        self.assertEqual(llama_cpp_loader._resolve_thinking_suppression(
            replace(spec, gguf_prompt_mode="embedded_template"), {}),
            ("embedded_template", "registry_embedded_template"))

    def test_invalid_mode_is_rejected(self):
        with patch.object(registry, "_get_registry_data", return_value={"models": {
            "bad": {"backend": "llama_cpp", "gguf_path": "local.gguf", "gguf_prompt_mode": "typo"}}}):
            with self.assertRaisesRegex(registry.RegistryError, "gguf_prompt_mode"):
                registry.get_model_spec("bad")

    def test_rendered_template_and_input_ids_are_forwarded_without_suffix_or_bos(self):
        for thinking in [False, True]:
            bundle, module, factory, formatter, expected = self.fixture(thinking=thinking)
            with patch.dict(sys.modules, {"llama_cpp.llama_chat_format": module}):
                result = generator.generate_prompt_candidates(bundle, "a traveler", "original", "negative",
                           {"enable_thinking": thinking, "seed_mode": "fixed", "llm_seed": 42})
            self.assertEqual(factory.call_args.kwargs["template"], "actual embedded template")
            self.assertEqual(formatter.call_args.kwargs["enable_thinking"], thinking)
            self.assertEqual(formatter.call_args.kwargs["messages"][1]["content"], "a traveler")
            bundle.model.tokenize.assert_called_once_with(expected.encode("utf-8"), add_bos=False, special=True)
            self.assertEqual(bundle.model.call_args.args[0], [11, 12, 13])
            self.assertEqual(bundle.model.call_args.kwargs["stop"], ["<custom-eos>"])
            bundle.model.create_chat_completion.assert_not_called()
            self.assertEqual(result.input_debug["raw_prompt_text"], expected)
            self.assertEqual(result.input_debug["tokenized_input_length"], 3)
            self.assertEqual(result.input_debug["input_mode"], "llama_cpp_embedded_template")
            self.assertFalse(result.input_debug["no_think_suffix_applied"])

    def test_user_authored_no_think_text_is_preserved_verbatim(self):
        request = "a traveler /no_think"
        bundle, module, _, formatter, _ = self.fixture(request)
        with patch.dict(sys.modules, {"llama_cpp.llama_chat_format": module}):
            result = generator.generate_prompt_candidates(bundle, request, "", "", {})
        self.assertEqual(formatter.call_args.kwargs["messages"][1]["content"], request)
        self.assertEqual(result.input_debug["raw_prompt_text"].count("/no_think"), 1)

    def test_missing_or_broken_template_never_falls_back_to_legacy_input(self):
        for missing in [True, False]:
            bundle, module, factory, _, _ = self.fixture()
            if missing:
                bundle.model.metadata = {}
            else:
                factory.side_effect = ValueError("invalid embedded template")
            with patch.dict(sys.modules, {"llama_cpp.llama_chat_format": module}):
                with self.assertRaises(generator.GenerationError):
                    generator.generate_prompt_candidates(bundle, "a traveler", "", "", {})
            bundle.model.assert_not_called()
            bundle.model.create_chat_completion.assert_not_called()

    def test_template_tokenization_memoryerror_is_classified(self):
        bundle, module, _, _, _ = self.fixture()
        bundle.model.tokenize.side_effect = MemoryError("native tokenization allocation failed")
        with patch.dict(sys.modules, {"llama_cpp.llama_chat_format": module}):
            with self.assertRaises(generator.GenerationError) as caught:
                generator.generate_prompt_candidates(bundle, "a traveler", "", "", {})
        self.assertTrue(caught.exception.is_oom)
        bundle.model.assert_not_called()
