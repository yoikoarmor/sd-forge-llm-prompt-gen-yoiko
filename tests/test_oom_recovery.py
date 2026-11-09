import types
import unittest
import weakref
from unittest.mock import patch
import test_generation_safety as safety
forge = safety.forge

class OomRecoveryTests(unittest.TestCase):

    def run_failure(self, *, phase='generate', load_mode='keep_loaded', debug=False, oom_type='runtime'):
        import torch
        probes = {'generate_calls': 0, 'cache_flushes': []}

        def raise_failure():
            temporary = torch.empty(8)
            probes['temporary'] = weakref.ref(temporary)
            if oom_type == 'typed':
                raise torch.OutOfMemoryError('allocation failed')
            if oom_type == 'ordinary':
                raise RuntimeError('unrelated generation failure')
            raise RuntimeError('CUDA out of memory during simulated ' + phase)

        class TransferFailure:

            def to(self, device):
                raise_failure()

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 9
            unk_token_id = -1
            chat_template = None

            def __call__(self, text, **kwargs):
                return {'input_ids': TransferFailure() if phase == 'transfer' else torch.tensor([[1, 2]])}

            def decode(self, tokens, **kwargs):
                return 'portrait, soft light'

            def convert_tokens_to_ids(self, token):
                return 9

        class Model:

            def __init__(self):
                self.weight = torch.empty(1)

            def parameters(self):
                yield self.weight

            def generate(self, **kwargs):
                probes['generate_calls'] += 1
                raise_failure()
        spec = forge.registry_module.ModelSpec('oom-test', 'local', device_map='cpu')
        runtime = forge.runtime_module.LlmRuntime()
        runtime._bundle = types.SimpleNamespace(spec=spec, model=Model(), tokenizer=Tokenizer(), load_debug={})
        runtime._signature = spec.signature()
        probes['model'] = weakref.ref(runtime._bundle.model)
        probes['weight'] = weakref.ref(runtime._bundle.model.weight)

        def empty_cache():
            probes['cache_flushes'].append({key: probes[key]() is None for key in ('model', 'weight', 'temporary')})
        script = forge.ForgeLlmPromptGenScript()
        script._log = lambda message: None
        original = '  keep this exact formatting, coat, coat  '
        decision = forge.build_initial_decision(True, spec.key, load_mode, 'auto', 128, 'portrait', original)
        with patch.object(forge, 'get_runtime', return_value=runtime), patch.object(forge, 'get_model_spec', return_value=spec), patch.object(forge, 'get_generation_defaults', return_value={'seed_mode': 'fixed', 'llm_seed': 123, 'debug_compare_input_variants': debug}), patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'device_count', return_value=0), patch.object(torch.cuda, 'manual_seed_all'), patch.object(torch.cuda, 'empty_cache', side_effect=empty_cache), patch.object(forge.traceback, 'print_exc'):
            result = script._generate_with_llm(decision, 'blur')
        self.assertEqual(result.final_prompt, original)
        self.assertFalse(result.changed)
        self.assertTrue(result.error)
        return (runtime, probes)

    def assert_released_before_flush(self, probes):
        self.assertTrue(probes['cache_flushes'], 'OOM recovery must flush the cache after releasing references')
        self.assertTrue(all((all(state.values()) for state in probes['cache_flushes'])), probes['cache_flushes'])
        self.assertIsNone(probes['model']())
        self.assertIsNone(probes['weight']())
        self.assertIsNone(probes['temporary']())

    def test_input_transfer_oom_unloads_even_in_keep_loaded_mode(self):
        (runtime, probes) = self.run_failure(phase='transfer')
        self.assertFalse(runtime.is_loaded())
        self.assertEqual(probes['generate_calls'], 0)
        self.assert_released_before_flush(probes)

    def test_generation_oom_releases_tracebacks_before_cache_flush(self):
        for mode in ('keep_loaded', 'load_then_unload_before_image_gen'):
            with self.subTest(mode=mode):
                (runtime, probes) = self.run_failure(load_mode=mode)
                self.assertFalse(runtime.is_loaded())
                self.assert_released_before_flush(probes)

    def test_typed_oom_does_not_depend_on_english_error_text(self):
        for phase in ('transfer', 'generate'):
            with self.subTest(phase=phase):
                (runtime, probes) = self.run_failure(phase=phase, oom_type='typed')
                self.assertFalse(runtime.is_loaded())
                self.assert_released_before_flush(probes)

    def test_debug_oom_aborts_before_other_variants_and_main_generation(self):
        (runtime, probes) = self.run_failure(debug=True)
        self.assertFalse(runtime.is_loaded())
        self.assertEqual(probes['generate_calls'], 1)
        self.assert_released_before_flush(probes)

    def test_non_oom_failure_keeps_requested_loaded_model(self):
        (runtime, probes) = self.run_failure(oom_type='ordinary')
        self.assertTrue(runtime.is_loaded())
        self.assertIsNotNone(probes['model']())
if __name__ == '__main__':
    unittest.main()
