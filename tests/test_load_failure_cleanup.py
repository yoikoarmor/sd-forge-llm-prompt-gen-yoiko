import types
import unittest
import weakref
from unittest.mock import patch
from backend import loader, registry, runtime
import test_generation_safety as safety

class LoadFailureCleanupTests(unittest.TestCase):

    def failing_loader(self, probes, error_class=loader.ModelLoadError, wrapped=True):
        import torch

        class PartialModel:
            pass

        def load_partial_model(*args, **kwargs):
            model = PartialModel()
            model.weight = torch.empty(1)
            temporary = torch.empty(8)
            probes.update(model=weakref.ref(model), weight=weakref.ref(model.weight), temporary=weakref.ref(temporary))
            if wrapped:
                try:
                    raise torch.OutOfMemoryError('simulated load allocation failed')
                except Exception as exc:
                    raise error_class('failed to load local model') from exc
            raise torch.OutOfMemoryError('simulated load allocation failed')
        return load_partial_model

    def cache_flush(self, probes, *, fail=False):

        def flush():
            probes.setdefault('flushes', []).append({key: probes[key]() is None for key in ('model', 'weight', 'temporary')})
            if fail:
                raise RuntimeError('simulated cache cleanup error')
        return flush

    def assert_released(self, probes):
        self.assertEqual(probes.get('flushes'), [{'model': True, 'weight': True, 'temporary': True}])

    def test_raw_and_wrapped_oom_release_partial_model_before_raising(self):
        import torch
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                probes = {}
                rt = runtime.LlmRuntime()
                spec = registry.ModelSpec('load-test', 'local', device_map='cpu')
                with patch.object(runtime, 'load_model_bundle', side_effect=self.failing_loader(probes, wrapped=wrapped)), patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'empty_cache', side_effect=self.cache_flush(probes)):
                    try:
                        rt.ensure_loaded(spec)
                    except Exception as exc:
                        self.assert_released(probes)
                        self.assertIsInstance(exc, loader.ModelLoadError)
                        self.assertIsNone(exc.__cause__)
                        self.assertIsNone(exc.__context__)
                        self.assertIn('load_partial_model', exc.load_traceback)
                        self.assertIn('simulated load allocation failed', exc.load_traceback)
                    else:
                        self.fail('Expected load failure')
                self.assertFalse(rt.is_loaded())
                self.assertIsNone(rt._signature)

    def test_failed_load_is_clean_before_following_gguf_load(self):
        import torch
        probes = {}
        rt = runtime.LlmRuntime()
        hf = registry.ModelSpec('hf', 'local', device_map='cpu')
        gguf = registry.ModelSpec('gguf', 'local.gguf', backend='llama_cpp')
        gguf_bundle = types.SimpleNamespace(spec=gguf, model=types.SimpleNamespace(is_alive=lambda : True), tokenizer=None, load_debug={})
        fail_load = self.failing_loader(probes)

        def load(spec, **kwargs):
            if spec.backend == 'transformers':
                return fail_load(spec)
            self.assert_released(probes)
            return gguf_bundle
        with patch.object(runtime, 'load_model_bundle', side_effect=load), patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'empty_cache', side_effect=self.cache_flush(probes)):
            try:
                rt.ensure_loaded(hf)
            except loader.ModelLoadError:
                self.assertEqual(rt.ensure_loaded(gguf), (gguf_bundle, True))
            else:
                self.fail('Expected HF load failure')
        self.assertTrue(rt.is_loaded())
        self.assertEqual(rt._signature, gguf.signature())

    def test_setup_error_in_keep_loaded_hook_flushes_unregistered_bundle(self):
        import torch
        forge = safety.forge
        probes = {}
        rt = forge.runtime_module.LlmRuntime()
        spec = forge.registry_module.ModelSpec('load-test', 'local', device_map='cpu')
        script = forge.ForgeLlmPromptGenScript()
        script._log = lambda message: None
        original = '  coat, coat  '
        decision = forge.build_initial_decision(True, spec.key, 'keep_loaded', 'auto', 128, 'portrait', original)
        with patch.object(forge, 'get_runtime', return_value=rt), patch.object(forge, 'get_model_spec', return_value=spec), patch.object(forge, 'get_generation_defaults', return_value={}), patch.object(forge.runtime_module, 'load_model_bundle', side_effect=self.failing_loader(probes, forge.ModelLoadError)), patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'empty_cache', side_effect=self.cache_flush(probes)):
            result = script._generate_with_llm(decision, 'blur')
        self.assert_released(probes)
        self.assertFalse(rt.is_loaded())
        self.assertEqual(result.final_prompt, original)
        self.assertEqual(result.error, 'failed to load local model')

    def test_missing_dependency_type_is_preserved_and_cleanup_failure_does_not_mask_it(self):
        import torch
        probes = {}
        rt = runtime.LlmRuntime()
        spec = registry.ModelSpec('load-test', 'local', device_map='cpu')
        with patch.object(runtime, 'load_model_bundle', side_effect=self.failing_loader(probes, loader.MissingDependencyError)), patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'empty_cache', side_effect=self.cache_flush(probes, fail=True)):
            with self.assertRaisesRegex(loader.MissingDependencyError, 'failed to load local model'):
                rt.ensure_loaded(spec)
        self.assert_released(probes)
if __name__ == '__main__':
    unittest.main()
