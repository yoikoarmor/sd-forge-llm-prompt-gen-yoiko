import ast
import hashlib
import inspect
import json
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import Mock, patch

import test_generation_safety as safety
from backend import generator, llama_cpp_loader, loader, prompt_builder, registry, runtime
from tools import configure_lora_family


class NeoRegistryTests(unittest.TestCase):
    def test_same_repository_explicit_tokenizer_revision_is_resolved_separately(self):
        for origin in ["base", "adapter"]:
            spec = registry.ModelSpec("test", "test/base", adapter_path="test/adapter",
                                      base_model_revision="base-revision", adapter_revision="adapter-revision",
                                      tokenizer_name_or_path="test/" + origin, tokenizer_revision="tokenizer-revision")
            base = loader.ResolvedReference("test/base", "huggingface", resolved_reference="base-snapshot")
            adapter = loader.ResolvedReference("test/adapter", "huggingface", resolved_reference="adapter-snapshot")
            separate = loader.ResolvedReference("test/" + origin, "huggingface", resolved_reference="tokenizer-snapshot")
            with patch.object(loader, "_resolve_reference", return_value=separate) as resolve:
                self.assertEqual(loader._resolve_tokenizer_reference(spec, base, adapter, {})[0], "tokenizer-snapshot")
                self.assertEqual(resolve.call_args.kwargs["revision"], "tokenizer-revision")
            spec.tokenizer_revision = origin + "-revision"
            with patch.object(loader, "_resolve_reference") as resolve:
                self.assertEqual(loader._resolve_tokenizer_reference(spec, base, adapter, {})[0], origin + "-snapshot")
                resolve.assert_not_called()
            spec.tokenizer_name_or_path = None
            spec.tokenizer_source = origin
            spec.tokenizer_revision = "tokenizer-revision"
            with patch.object(loader, "_resolve_reference", return_value=separate) as resolve:
                result = loader._resolve_tokenizer_reference(spec, base, adapter, {"tokenizer.json": {"exists": True}})
                self.assertEqual(result[0], "tokenizer-snapshot")
                self.assertEqual(resolve.call_args.kwargs["revision"], "tokenizer-revision")

    def test_native_example_and_user_merge_overrides_and_disabled_models(self):
        example = json.loads(registry.MODEL_REGISTRY_EXAMPLE_PATH.read_text(encoding="utf-8"))
        native_gguf = [k for k, v in example["models"].items()
                       if v.get("enabled", True) and v.get("backend") == "llama_cpp"]
        self.assertEqual(len(native_gguf), 4)
        with tempfile.TemporaryDirectory() as tmp:
            user = Path(tmp) / "user.json"
            with patch.object(registry, "MODEL_REGISTRY_PATH", user):
                self.assertTrue(set(native_gguf).issubset(registry.get_ui_model_choices()))
                user.write_text(json.dumps({"models": {
                    native_gguf[0]: {"n_ctx": 8192}, native_gguf[1]: {"enabled": False},
                    "local": {"backend": "llama_cpp", "gguf_path": "./local.gguf",
                              "gguf_lora_path": "./adapter.gguf", "n_gpu_layers": 17,
                              "n_batch": 64, "n_threads": 3, "flash_attn": False}
                }}), encoding="utf-8")
                spec = registry.get_model_spec(native_gguf[0])
                self.assertEqual(spec.n_ctx, 8192)
                self.assertEqual(spec.gguf_repo_id, example["models"][native_gguf[0]]["gguf_repo_id"])
                self.assertNotIn(native_gguf[1], registry.get_ui_model_choices())
                local = registry.get_model_spec("local")
                self.assertEqual((local.n_gpu_layers, local.n_batch, local.n_threads, local.flash_attn),
                                 (17, 64, 3, False))
                self.assertTrue(local.gguf_path.endswith("local.gguf"))
                self.assertTrue(local.gguf_lora_path.endswith("adapter.gguf"))

    def test_configure_native_gguf_preserves_existing_and_requires_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "registry.json"
            target.write_text('{"models":{"custom":{"n_ctx":1234}}}', encoding="utf-8")
            before = target.read_bytes()
            with self.assertRaisesRegex(ValueError, "gguf-revision"):
                configure_lora_family.configure(target, gguf_model="test/model")
            self.assertEqual(target.read_bytes(), before)
            configure_lora_family.configure(target, gguf_model="test/model", gguf_revision="a" * 40)
            data = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(data["models"]["custom"], {"n_ctx": 1234})
            entry = data["models"]["qwen3.5-9b-lora-7-q4km"]
            self.assertEqual(entry["gguf_repo_id"], "test/model")
            self.assertEqual(entry["gguf_revision"], "a" * 40)
            self.assertIsNone(entry["gguf_path"])
            self.assertNotIn("llama_server_path", entry)

    def test_native_gguf_hash_checked_before_load_and_adapter_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "model.gguf"
            model_file.write_bytes(b"native gguf fixture")
            spec = registry.ModelSpec("native", backend="llama_cpp", gguf_path=str(model_file),
                                      gguf_sha256="0" * 64, gguf_lora_path="adapter.gguf")
            llama = Mock(return_value=types.SimpleNamespace(metadata={}))
            with patch.object(llama_cpp_loader, "_import_llama_cpp", return_value=llama):
                with self.assertRaisesRegex(loader.ModelLoadError, "SHA256 mismatch"):
                    llama_cpp_loader.load_llama_cpp_bundle(spec)
                llama.assert_not_called()
                spec.gguf_sha256 = hashlib.sha256(model_file.read_bytes()).hexdigest()
                bundle = llama_cpp_loader.load_llama_cpp_bundle(spec)
            self.assertEqual(llama.call_args.kwargs["lora_path"], "adapter.gguf")
            self.assertEqual(bundle.load_debug["backend"], "llama_cpp")

    def test_native_gguf_remote_revision_forwarded(self):
        spec = registry.ModelSpec("native", backend="llama_cpp", gguf_repo_id="test/repo",
                                  gguf_filename="model.gguf", gguf_revision="a" * 40)
        with patch("huggingface_hub.hf_hub_download", return_value="cached.gguf") as download:
            path, source, downloaded = llama_cpp_loader._resolve_gguf_path(spec)
        self.assertEqual(path, "cached.gguf")
        self.assertEqual(download.call_args.kwargs["revision"], "a" * 40)

    def test_native_runtime_reuses_and_closes_without_process_protocol(self):
        spec = registry.ModelSpec("native", backend="llama_cpp")
        model = types.SimpleNamespace(close=Mock())
        bundle = types.SimpleNamespace(spec=spec, model=model, tokenizer=None, load_debug={})
        rt = runtime.LlmRuntime()
        with patch.object(runtime, "load_model_bundle", return_value=bundle) as load:
            self.assertEqual(rt.ensure_loaded(spec), (bundle, True))
            self.assertEqual(rt.ensure_loaded(spec), (bundle, False))
            self.assertTrue(rt.is_loaded())
            rt.unload()
            load.assert_called_once()
            model.close.assert_called_once()


class NeoGenerationTests(unittest.TestCase):
    def test_untagged_reasoning_retains_native_rejection(self):
        for output in ["Thinking process: analyze the request, select the subject",
                       "**Task: portrait, **Constraints: only positive tags"]:
            self.assertEqual(prompt_builder.is_generated_prompt_strong_enough(output),
                             (False, "reasoning_text_detected"))

    def test_native_gguf_initial_memoryerror_unloads_keep_loaded_model(self):
        forge = safety.forge
        for raw in [True, False]:
            close = Mock()
            class Model:
                def __call__(self, *args, **kwargs):
                    raise MemoryError("native allocation failed")
                create_chat_completion = __call__
            model = Model()
            model.close = close
            spec = forge.registry_module.ModelSpec("native", backend="llama_cpp")
            bundle = types.SimpleNamespace(spec=spec, model=model, tokenizer=None,
                         load_debug={"backend": "llama_cpp", "thinking_suppression_resolved": "no_think" if raw else "none"})
            rt = forge.runtime_module.LlmRuntime()
            rt._bundle, rt._signature = bundle, spec.signature()
            script = forge.ForgeLlmPromptGenScript()
            script._log = lambda text: None
            decision = forge.build_initial_decision(True, "native", "keep_loaded", "auto", 128,
                                                     "portrait", "  original, exact  ")
            with patch.object(forge, "get_runtime", return_value=rt), \
                 patch.object(forge, "get_model_spec", return_value=spec), \
                 patch.object(forge, "get_generation_defaults", return_value={}):
                result = script._generate_with_llm(decision, "")
            self.assertFalse(rt.is_loaded())
            close.assert_called_once()
            self.assertEqual(result.runtime_action, "unloaded_after_oom")
            self.assertEqual(result.final_prompt, "  original, exact  ")

    def test_native_gguf_explicit_top_k_zero_is_preserved(self):
        self.assertEqual(generator._build_llama_cpp_generation_kwargs({"top_k": 0}, 42)["top_k"], 0)

    def test_multiple_hf_candidates_and_greedy_mode(self):
        import torch
        for sample in [True, False]:
            bundle = safety.GenerationSafetyTests.make_bundle(types.SimpleNamespace(torch=torch))
            seen = {}
            def generate(**kwargs):
                seen.update(kwargs)
                return torch.tensor([[1, 2, 3], [1, 2, 4], [1, 2, 5]])
            bundle.model.generate = generate
            bundle.tokenizer.decode = lambda tokens, **kwargs: f"candidate {int(tokens[-1])}, soft light"
            result = generator.generate_prompt_candidates(bundle, "portrait", "coat", "blur",
                      {"do_sample": sample, "seed_mode": "fixed", "llm_seed": 123}, num_candidates=3)
            self.assertEqual(len(result.candidates), 3)
            self.assertEqual(result.chosen_candidate, "candidate 3, soft light")
            self.assertEqual(seen["num_return_sequences"], 3)
            if not sample:
                self.assertEqual(seen["num_beams"], 3)

    def test_native_gguf_stop_during_raw_and_chat_stream_discards_candidates(self):
        for raw in [True, False]:
            state = {"stop": False, "closed": False, "calls": 0}
            def stream(*args, **kwargs):
                state["calls"] += 1
                try:
                    yield {"choices": [{"text": "portrait, soft light"}]}
                    state["stop"] = True
                    yield {"choices": [{"delta": {"content": ", unfinished"}}]}
                finally:
                    state["closed"] = True
            class Model:
                __call__ = staticmethod(stream)
                create_chat_completion = staticmethod(stream)
            bundle = types.SimpleNamespace(model=Model(), load_debug={"backend": "llama_cpp",
                    "thinking_suppression_resolved": "no_think" if raw else "none"})
            result = generator.generate_prompt_candidates(bundle, "portrait", "coat", "blur", {},
                       interrupt_checker=lambda: state["stop"], num_candidates=3)
            self.assertTrue(result.interrupted)
            self.assertEqual(result.candidates, [])
            self.assertEqual(result.selected_positive, "")
            self.assertTrue(state["closed"])
            self.assertEqual(state["calls"], 1)

    def test_native_gguf_stream_oom_is_classified(self):
        def stream(**kwargs):
            yield {"choices": [{"delta": {"content": "portrait"}}]}
            raise MemoryError("native allocation failed")
        bundle = types.SimpleNamespace(model=types.SimpleNamespace(create_chat_completion=stream),
                                       load_debug={"backend": "llama_cpp"})
        with self.assertRaises(generator.GenerationError) as caught:
            generator.generate_prompt_candidates(bundle, "portrait", "", "", {}, interrupt_checker=lambda: False)
        self.assertTrue(caught.exception.is_oom)

    def test_native_gguf_candidates_keep_seed_increment_and_thinking_prefix(self):
        seen = []
        class Model:
            def __call__(self, prompt, **kwargs):
                seen.append((prompt, kwargs))
                return iter([{"choices": [{"text": f"portrait {kwargs['seed']}, soft light"}]}])
        bundle = types.SimpleNamespace(model=Model(), load_debug={"backend": "llama_cpp",
                                       "thinking_suppression_resolved": "no_think"})
        result = generator.generate_prompt_candidates(bundle, "portrait", "coat", "blur",
                    {"seed_mode": "fixed", "llm_seed": 42}, num_candidates=2)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual([item[1]["seed"] for item in seen], [42, 43])
        self.assertTrue(all("<think>\n\n</think>" in item[0] for item in seen))


class NeoHookAndCompatibilityTests(unittest.TestCase):
    def test_actual_ui_return_contract_and_hook_order(self):
        forge = safety.forge
        tree = ast.parse(inspect.getsource(forge))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ForgeLlmPromptGenScript")
        ui = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "ui")
        returned = next(n for n in ui.body if isinstance(n, ast.Return))
        expected = ["llm_enabled", "llm_model_name", "llm_load_mode", "llm_weight_mode",
                    "llm_max_new_tokens", "llm_num_candidates", "llm_gen_prompt", "llm_selected_candidate"]
        self.assertEqual([n.id for n in returned.value.elts], expected)
        for name in ["before_process", "process"]:
            self.assertEqual(list(inspect.signature(getattr(forge.ForgeLlmPromptGenScript, name)).parameters)[2:], expected)

    def test_preview_and_selected_candidate_keep_native_controls(self):
        forge = safety.forge
        script = forge.ForgeLlmPromptGenScript()
        script._log = lambda text: None
        rt = Mock()
        rt.ensure_loaded.return_value = (Mock(), True)
        spec = forge.registry_module.ModelSpec("native", "local")
        result = forge.generator_module.GenerationResult(llm_called=True,
                    candidates=["first, soft light", "second, soft light"],
                    chosen_candidate="first, soft light", selected_positive="first, soft light")
        decision = forge.build_initial_decision(True, "native", "load_then_unload_before_image_gen",
                    "4bit", 256, "portrait", "coat", 2, "second, soft light")
        with patch.object(forge, "get_runtime", return_value=rt), \
             patch.object(forge, "get_model_spec", return_value=spec), \
             patch.object(forge, "get_generation_defaults", return_value={}), \
             patch.object(forge, "generate_prompt_candidates", return_value=result) as generate, \
             patch.object(script, "_offload_current_image_model_to_ram", return_value=True) as offload:
            updated = script._generate_with_llm(decision, "blur", offload_image_model=True)
            self.assertEqual(updated.final_prompt, "second, soft light, coat")
            self.assertTrue(updated.image_model_offloaded_to_ram)
            self.assertEqual(generate.call_args.kwargs["num_candidates"], 2)
            self.assertEqual(generate.call_args.kwargs["generation_defaults"]["max_new_tokens"], 225)
            self.assertTrue(rt.ensure_loaded.call_args.args[0].load_in_4bit)
            self.assertEqual(rt.ensure_loaded.call_args.args[0].runtime_weight_mode, "4bit")
            offload.assert_called_once()
            offload.reset_mock()
            script._preview_candidates(True, "native", "keep_loaded", "auto", 384, 3, "portrait")
            offload.assert_not_called()
            self.assertEqual(generate.call_args.kwargs["num_candidates"], 3)

    def test_auto_high_vram_throughput_profile_preserved(self):
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        spec = registry.ModelSpec("qwen", "Qwen/Qwen3.5-9B", adapter_path="local")
        with patch.object(loader, "_get_total_vram_bytes", return_value=48 * 1024**3):
            effective, debug = loader._choose_effective_spec_for_throughput(spec, torch)
        self.assertFalse(effective.load_in_4bit)
        self.assertTrue(effective.merge_lora_for_inference)
        self.assertTrue(debug["throughput_profile_applied"])

    def test_failed_throughput_profile_is_collected_before_nf4_retry(self):
        import torch
        probes = {}
        calls = []
        spec = registry.ModelSpec("qwen", "local", device_map="cpu")
        expected = object()

        def load_once(original, effective, debug, logger=None):
            calls.append(debug)
            if len(calls) == 1:
                temporary = torch.empty(8)
                probes["temporary"] = weakref.ref(temporary)
                raise loader.ModelLoadError("out of memory during BF16 profile")
            self.assertIsNone(probes["temporary"]())
            self.assertEqual(probes["flushed"], [True])
            self.assertTrue(debug["throughput_profile_fallback_to_original"])
            return expected

        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: probes.setdefault("flushed", []).append(probes["temporary"]() is None)))
        with patch.object(loader, "_import_torch", return_value=fake_torch), \
             patch.object(loader, "_choose_effective_spec_for_throughput", return_value=(spec, {"throughput_profile_applied": True})), \
             patch.object(loader, "_load_model_bundle_once", side_effect=load_once):
            self.assertIs(loader.load_model_bundle(spec), expected)

    def test_unsupported_local_text_only_architecture_fails_before_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text('{"model_type":"qwen3_5_text"}', encoding="utf-8")
            with patch("transformers.models.auto.configuration_auto.CONFIG_MAPPING_NAMES", {}):
                with self.assertRaisesRegex(loader.ModelLoadError, "separate compatible environment"):
                    loader._check_local_model_architecture(tmp)
