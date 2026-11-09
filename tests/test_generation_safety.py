import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import generator

def load_forge_script():
    from backend.compat import apply_runtime_compat_shims
    apply_runtime_compat_shims()
    module_name = 'test_forge_llm_script'
    spec = importlib.util.spec_from_file_location(module_name, ROOT / 'scripts' / 'forge_llm_prompt_gen.py')
    module = importlib.util.module_from_spec(spec)
    modules = types.ModuleType('modules')
    modules.scripts = types.SimpleNamespace(Script=object, AlwaysVisible=True)
    modules.shared = types.SimpleNamespace(state=types.SimpleNamespace(interrupted=False, stopping_generation=False))
    with patch.dict(sys.modules, {'gradio': types.ModuleType('gradio'), 'modules': modules, module_name: module}):
        spec.loader.exec_module(module)
    return module
forge = load_forge_script()

class GenerationSafetyTests(unittest.TestCase):

    def setUp(self):
        import torch
        self.torch = torch

    def make_bundle(self):
        torch = self.torch

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 9
            unk_token_id = -1
            chat_template = None

            def __call__(self, text, **kwargs):
                return {'input_ids': torch.tensor([[1, 2]])}

            def decode(self, tokens, **kwargs):
                return 'a peaceful landscape, soft natural light'

            def convert_tokens_to_ids(self, token):
                return 9

        class Model:

            def parameters(self):
                yield torch.empty(0)

            def generate(self, input_ids, **kwargs):
                torch.rand(3)
                return torch.cat((input_ids, torch.tensor([[3, 4]])), dim=1)
        return types.SimpleNamespace(model=Model(), tokenizer=Tokenizer(), load_debug={})

    def run_generation(self, bundle=None, **kwargs):
        defaults = {'seed_mode': 'fixed', 'llm_seed': 123, **kwargs.pop('generation_defaults', {})}
        return generator.generate_prompt_candidates(bundle or self.make_bundle(), 'landscape', 'detail', 'blur', defaults, **kwargs)

    def test_zero_pad_token_is_preserved(self):
        kwargs = generator._build_generation_kwargs(self.make_bundle().tokenizer, {})
        self.assertEqual(kwargs['pad_token_id'], 0)

    def test_rng_restored_after_generation(self):
        torch = self.torch
        torch.manual_seed(9876)
        before = torch.random.get_rng_state().clone()
        with patch.object(torch.cuda, 'is_available', return_value=False):
            self.run_generation()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_rng_restored_after_failed_generation(self):
        torch = self.torch
        torch.manual_seed(9876)
        before = torch.random.get_rng_state().clone()
        bundle = self.make_bundle()

        def fail(**kwargs):
            torch.rand(1)
            raise RuntimeError('test failure')
        bundle.model.generate = fail
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaises(generator.GenerationError):
                self.run_generation(bundle)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_stop_during_generation_discards_partial_output(self):
        state = {'stopped': False, 'steps': 0}
        bundle = self.make_bundle()

        def generate(input_ids, **kwargs):
            for i in range(10):
                state['steps'] += 1
                state['stopped'] = i >= 2
                criteria = kwargs.get('stopping_criteria', [])
                if any((bool(stop(input_ids, None).all()) for stop in criteria)):
                    break
            return self.torch.cat((input_ids, self.torch.tensor([[3, 4]])), dim=1)
        bundle.model.generate = generate
        with patch.object(self.torch.cuda, 'is_available', return_value=False):
            result = self.run_generation(bundle, interrupt_checker=lambda : state['stopped'])
        self.assertLessEqual(state['steps'], 3)
        self.assertTrue(result.interrupted)
        self.assertEqual(result.selected_positive, '')
        self.assertEqual(result.candidates, [])

    def test_debug_comparison_does_not_change_main_result(self):
        bundle = self.make_bundle()
        samples = []

        def generate(input_ids, **kwargs):
            samples.append(float(self.torch.rand(1)))
            return self.torch.cat((input_ids, self.torch.tensor([[3, 4]])), dim=1)
        bundle.model.generate = generate
        with patch.object(self.torch.cuda, 'is_available', return_value=False):
            self.run_generation(bundle, generation_defaults={'debug_compare_input_variants': True})
        self.assertEqual(len(samples), 4)
        self.assertEqual(len(set(samples)), 1)

    def test_rng_restores_available_gpu_devices(self):
        torch = self.torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        cpu_before = torch.random.get_rng_state().clone()
        gpu_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
        self.run_generation()
        self.assertTrue(torch.equal(cpu_before, torch.random.get_rng_state()))
        for (before, after) in zip(gpu_before, torch.cuda.get_rng_state_all()):
            self.assertTrue(torch.equal(before, after))

    def test_stop_during_debug_comparison_prevents_main_generation(self):
        state = {'stopped': False, 'calls': 0}
        bundle = self.make_bundle()

        def generate(input_ids, **kwargs):
            state['calls'] += 1
            state['stopped'] = True
            for stop in kwargs.get('stopping_criteria', []):
                stop(input_ids, None)
            return self.torch.cat((input_ids, self.torch.tensor([[3, 4]])), dim=1)
        bundle.model.generate = generate
        with patch.object(self.torch.cuda, 'is_available', return_value=False):
            result = self.run_generation(bundle, generation_defaults={'debug_compare_input_variants': True}, interrupt_checker=lambda : state['stopped'])
        self.assertEqual(state['calls'], 1)
        self.assertTrue(result.interrupted)
        self.assertEqual(result.selected_positive, '')

    def test_preexisting_stop_does_not_generate(self):
        bundle = self.make_bundle()
        bundle.model.generate = Mock()
        with patch.object(self.torch.cuda, 'is_available', return_value=False):
            result = self.run_generation(bundle, interrupt_checker=lambda : True)
        bundle.model.generate.assert_not_called()
        self.assertTrue(result.interrupted)
        self.assertFalse(result.llm_called)

    def test_none_pad_token_falls_back_to_eos(self):
        tokenizer = self.make_bundle().tokenizer
        tokenizer.pad_token_id = None
        self.assertEqual(generator._build_generation_kwargs(tokenizer, {})['pad_token_id'], 9)

class ForgeHookTests(unittest.TestCase):

    def setUp(self):
        self.script = forge.ForgeLlmPromptGenScript()
        self.runtime = Mock()
        self.runtime.unload.return_value = True
        self.runtime_patcher = patch.object(forge, 'get_runtime', return_value=self.runtime)
        self.runtime_patcher.start()
        self.addCleanup(self.runtime_patcher.stop)
        self.script._log = Mock()

    def processing(self, prompt, negative='blur'):
        return types.SimpleNamespace(prompt=prompt, negative_prompt=negative, extra_generation_params={})

    def test_manual_batch_preserves_list_and_each_prompt(self):
        p = self.processing(['red coat', 'blue coat'], ['blur', 'grain'])
        self.script.before_process(p, True, 'none', 'keep_loaded', 'auto', 128, 1, 'portrait', '')
        self.assertEqual(p.prompt, ['portrait, red coat', 'portrait, blue coat'])
        self.assertEqual(p.negative_prompt, ['blur', 'grain'])

    def test_disabled_batch_is_unchanged(self):
        p = self.processing(['red coat', 'blue coat'])
        self.script.before_process(p, False, 'none', 'keep_loaded', 'auto', 128, 1, 'portrait', '')
        self.assertEqual(p.prompt, ['red coat', 'blue coat'])
        self.assertEqual(p._llm_prompt_gen_decision.original_prompt, ['red coat', 'blue coat'])

    def test_empty_gen_prompt_batch_is_unchanged(self):
        p = self.processing(['red coat', 'blue coat'])
        self.script.before_process(p, True, 'any-model', 'keep_loaded', 'auto', 128, 1, '', '')
        self.assertEqual(p.prompt, ['red coat', 'blue coat'])
        self.assertEqual(p._llm_prompt_gen_decision.original_prompt, ['red coat', 'blue coat'])

    def test_unload_mode_without_llm_call_releases_previous_model(self):
        for (model, prompt) in [('none', 'portrait'), ('any-model', '')]:
            with self.subTest(model=model):
                self.runtime.reset_mock()
                p = self.processing('coat')
                self.script.before_process(p, True, model, 'load_then_unload_before_image_gen', 'auto', 128, 1, prompt, '')
                self.runtime.unload.assert_called_once()

    def test_manual_keep_loaded_does_not_override_user_mode(self):
        self.script.before_process(self.processing('coat'), True, 'none', 'keep_loaded', 'auto', 128, 1, 'portrait', '')
        self.runtime.unload.assert_not_called()

    def test_llm_batch_reuses_generation_for_same_negative(self):
        p = self.processing(['red coat', 'blue coat'])

        def generate(decision, negative, **kwargs):
            decision.llm_called = True
            decision.selected_positive = 'a portrait, soft light'
            decision.final_prompt = decision.selected_positive + ', ' + decision.original_prompt
            decision.changed = True
            return decision
        with patch.object(self.script, '_generate_with_llm', side_effect=generate) as call:
            self.script.before_process(p, True, 'any-model', 'keep_loaded', 'auto', 128, 1, 'portrait', '')
        self.assertEqual(p.prompt, ['a portrait, soft light, red coat', 'a portrait, soft light, blue coat'])
        self.assertEqual(call.call_count, 1)

    def test_llm_batch_keeps_distinct_negative_references_separate(self):
        p = self.processing(['red coat', 'blue coat'], ['blur', 'grain'])
        with patch.object(self.script, '_generate_with_llm', side_effect=lambda decision, negative, **kwargs: decision) as call:
            self.script.before_process(p, True, 'any-model', 'keep_loaded', 'auto', 128, 1, 'portrait', '')
        self.assertEqual([item.args[1] for item in call.call_args_list], ['blur', 'grain'])
        self.assertEqual(p.prompt, ['red coat', 'blue coat'])

    def test_scalar_prompt_with_negative_batch_is_broadcast_correctly(self):
        p = self.processing('coat', ['blur', 'grain'])

        def generate(decision, negative, **kwargs):
            decision.llm_called = True
            decision.selected_positive = 'portrait, ' + negative
            self.script._apply_selected_positive(decision)
            return decision
        with patch.object(self.script, '_generate_with_llm', side_effect=generate):
            self.script.before_process(p, True, 'any-model', 'keep_loaded', 'auto', 128, 1, 'portrait', '')
        self.assertEqual(p.prompt, ['portrait, blur, coat', 'portrait, grain, coat'])
        self.assertEqual(p.negative_prompt, ['blur', 'grain'])

    def test_failed_batch_generation_leaves_all_originals_intact(self):
        p = self.processing(['red coat', 'blue coat'])

        def fail(decision, negative, **kwargs):
            decision.error = 'test load failure'
            return decision
        with patch.object(self.script, '_generate_with_llm', side_effect=fail):
            self.script.before_process(p, True, 'any-model', 'keep_loaded', 'auto', 128, 1, 'portrait', '')
        self.assertEqual(p.prompt, ['red coat', 'blue coat'])

    def test_interrupted_generation_does_not_clean_original(self):
        original = '  keep this exact formatting, Prompt:  '
        p = self.processing(original)
        result = types.SimpleNamespace(llm_called=True, candidates=[], candidate_debug=[], chosen_candidate='', chosen_candidate_debug={}, selected_positive='', fallback_used=False, fallback_reason='', llm_seed=123, seed_mode='fixed', interrupted=True, input_debug={})
        with patch.object(forge, 'get_model_spec', return_value=forge.registry_module.ModelSpec('any-model', 'local')), patch.object(forge, 'get_generation_defaults', return_value={}), patch.object(forge, 'generate_prompt_candidates', return_value=result):
            self.runtime.ensure_loaded.return_value = (Mock(), True)
            self.script.before_process(p, True, 'any-model', 'load_then_unload_before_image_gen', 'auto', 128, 1, 'portrait', '')
        self.assertEqual(p.prompt, original)
        self.assertTrue(p._llm_prompt_gen_decision.interrupted)
        self.runtime.unload.assert_called_once()

    def test_preexisting_stop_still_unloads_for_requested_mode(self):
        decision = forge.build_initial_decision(True, 'any-model', 'load_then_unload_before_image_gen', 'auto', 128, 'portrait', 'coat')
        with patch.object(self.script, '_is_interrupted', return_value=True):
            result = self.script._generate_with_llm(decision, 'blur')
        self.assertTrue(result.interrupted)
        self.runtime.ensure_loaded.assert_not_called()
        self.runtime.unload.assert_called_once()

    def test_scalar_manual_path_and_metadata_remain_compatible(self):
        p = self.processing('coat, portrait')
        self.script.before_process(p, True, 'none', 'keep_loaded', 'auto', 128, 1, 'portrait', '')
        self.script.process(p, True, 'none', 'keep_loaded', 'auto', 128, 1, 'portrait', '')
        self.assertEqual(p.prompt, 'portrait, coat')
        self.assertEqual(p.extra_generation_params['LLM Prompt Gen'], 'manual prepend')

    def test_direct_initial_decision_preserves_batch(self):
        result = forge.build_initial_decision(True, 'none', 'keep_loaded', 'auto', 128, 'portrait', ['red', 'blue'])
        self.assertEqual(result.original_prompt, ['red', 'blue'])
        self.assertEqual(result.final_prompt, ['portrait, red', 'portrait, blue'])
if __name__ == '__main__':
    unittest.main()
