import types
import unittest
from unittest.mock import Mock, patch
import test_generation_safety as safety
from backend import prompt_builder
forge = safety.forge
generator = safety.generator

class GenerationRegressionTests(unittest.TestCase):
    make_bundle = safety.GenerationSafetyTests.make_bundle

    def setUp(self):
        import torch
        self.torch = torch

    def test_explicit_zero_top_k_disables_top_k_filtering(self):
        kwargs = generator._build_generation_kwargs(self.make_bundle().tokenizer, {'top_k': 0})
        self.assertEqual(kwargs.get('top_k'), 0)

    def test_chat_template_does_not_duplicate_special_tokens(self):
        bundle = self.make_bundle()
        tokenizer = Mock(wraps=bundle.tokenizer)
        tokenizer.chat_template = 'test template'
        tokenizer.apply_chat_template = Mock(return_value='<bos>USER: portrait ASSISTANT:')
        generator._prepare_inputs(tokenizer, bundle.model, messages=[{'role': 'user', 'content': 'portrait'}])
        self.assertFalse(tokenizer.call_args.kwargs.get('add_special_tokens', True))

    def test_interrupted_result_keeps_resolved_seed_metadata(self):
        result = generator.generate_prompt_candidates(self.make_bundle(), 'portrait', 'coat', 'blur', {'seed_mode': 'fixed', 'llm_seed': 123}, interrupt_checker=lambda : True)
        self.assertTrue(result.interrupted)
        self.assertEqual((result.llm_seed, result.seed_mode), (123, 'fixed'))

    def test_unfinished_thinking_never_becomes_a_positive_prompt(self):
        text = '<think>Let me consider the subject, evaluate the composition, decide the lighting'
        self.assertEqual(prompt_builder.clean_generated_positive(text), '')
        self.assertFalse(prompt_builder.is_generated_prompt_strong_enough(text)[0])

    def test_reasoning_started_by_template_is_removed(self):
        text = 'Let me consider the subject, decide lighting</think>portrait, soft light'
        self.assertEqual(prompt_builder.clean_generated_positive(text), 'portrait, soft light')

class ForgeRegressionTests(unittest.TestCase):

    def setUp(self):
        self.script = forge.ForgeLlmPromptGenScript()
        self.script._log = Mock()
        self.runtime = Mock()
        patcher = patch.object(forge, 'get_runtime', return_value=self.runtime)
        patcher.start()
        self.addCleanup(patcher.stop)

    def processing(self, prompt, negative='blur'):
        return types.SimpleNamespace(prompt=prompt, negative_prompt=negative, extra_generation_params={})

    def test_empty_manual_gen_prompt_preserves_exact_original(self):
        originals = ['  Prompt: red coat, red coat  ', "'blue coat'"]
        for original in [originals, originals[0]]:
            with self.subTest(original=original):
                p = self.processing(original)
                self.script.before_process(p, True, 'none', 'keep_loaded', 'auto', 128, 1, '   ', '')
                self.assertEqual(p.prompt, original)
                self.assertFalse(p._llm_prompt_gen_decision.changed)

    def test_disabled_invalid_batch_still_unloads_previous_model(self):
        p = self.processing(['red', 'blue'], ['blur'])
        with patch.object(forge.traceback, 'print_exc') as logged:
            self.script.before_process(p, False, 'none', 'keep_loaded', 'auto', 128, 1, 'portrait', '')
        logged.assert_called_once()
        self.runtime.unload.assert_called_once()
        self.assertEqual(p.prompt, ['red', 'blue'])

    def test_invalid_batch_still_honors_unload_mode(self):
        p = self.processing(['red', 'blue'], ['blur'])
        with patch.object(forge.traceback, 'print_exc') as logged:
            self.script.before_process(p, True, 'any-model', 'load_then_unload_before_image_gen', 'auto', 128, 1, 'portrait', '')
        logged.assert_called_once()
        self.runtime.unload.assert_called_once()
        self.runtime.ensure_loaded.assert_not_called()
        self.assertEqual(p.prompt, ['red', 'blue'])

    def test_stop_after_first_result_discards_batch_and_prevents_more_calls(self):
        for negatives in [['blur', 'blur'], ['blur', 'grain']]:
            with self.subTest(negatives=negatives):
                state = {'stopped': False}
                originals = ['  red coat  ', 'blue coat']
                p = self.processing(originals, negatives)

                def generate(decision, negative, **kwargs):
                    decision.llm_called = True
                    decision.selected_positive = 'portrait, soft light'
                    self.script._apply_selected_positive(decision)
                    state['stopped'] = True
                    return decision
                with patch.object(self.script, '_is_interrupted', side_effect=lambda : state['stopped']), patch.object(self.script, '_generate_with_llm', side_effect=generate) as call:
                    self.script.before_process(p, True, 'any-model', 'keep_loaded', 'auto', 128, 1, 'portrait', '')
                self.assertEqual(p.prompt, originals)
                self.assertEqual(call.call_count, 1)
                self.assertTrue(p._llm_prompt_gen_decision.interrupted)
                self.assertFalse(p._llm_prompt_gen_decision.changed)

    def test_later_interrupt_is_reflected_in_batch_decision(self):
        p = self.processing(['red coat', 'blue coat', 'green coat'], ['blur', 'grain', 'blur'])

        def generate(decision, negative, **kwargs):
            decision.llm_called = True
            if negative == 'grain':
                decision.interrupted = True
            else:
                decision.selected_positive = 'portrait, soft light'
                self.script._apply_selected_positive(decision)
            return decision
        with patch.object(self.script, '_generate_with_llm', side_effect=generate) as call:
            self.script.before_process(p, True, 'any-model', 'keep_loaded', 'auto', 128, 1, 'portrait', '')
        self.assertEqual(p.prompt, ['red coat', 'blue coat', 'green coat'])
        self.assertEqual(call.call_count, 2)
        self.assertTrue(p._llm_prompt_gen_decision.interrupted)
        self.assertTrue(all((item.interrupted for item in p._llm_prompt_gen_decisions)))
        self.script.process(p, True, 'any-model', 'keep_loaded', 'auto', 128, 1, 'portrait', '')
        self.assertEqual(p.extra_generation_params['LLM Prompt Gen'], 'llm interrupted')
        self.assertTrue(p.extra_generation_params['LLM Interrupted'])
        self.assertNotIn('LLM Selected Positive', p.extra_generation_params)
if __name__ == '__main__':
    unittest.main()
