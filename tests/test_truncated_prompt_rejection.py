import types
import unittest
from unittest.mock import patch
import test_generation_safety as safety

from backend import generator, prompt_builder

class TruncatedPromptRejectionTests(unittest.TestCase):

    def test_unbalanced_generated_attention_or_schedule_is_rejected(self):
        for text in ['portrait, (bravery:1', 'portrait, (bravery:1.2', 'portrait, (', 'portrait, [summer:winter:0.5', 'portrait), soft light', 'portrait, (red] hair)']:
            with self.subTest(text=text):
                (usable, reason) = prompt_builder.is_generated_prompt_strong_enough(text)
                self.assertFalse(usable)
                self.assertEqual(reason, 'unbalanced_prompt_syntax')

    def test_valid_nested_and_escaped_syntax_remains_usable(self):
        for text in ['portrait, (bravery:1.2)', '((red hair, blue eyes:1.2):1.1), soft light', 'portrait, [summer:winter:0.5]', 'portrait, literal \\(bravery']:
            with self.subTest(text=text):
                self.assertEqual(prompt_builder.is_generated_prompt_strong_enough(text), (True, ''))

    def test_transformers_fallback_cannot_extend_unfinished_group_into_original(self):
        import torch
        helper = types.SimpleNamespace(torch=torch)
        bundle = safety.GenerationSafetyTests.make_bundle(helper)
        bundle.tokenizer.decode = lambda *args, **kwargs: 'portrait, (bravery:1'
        with patch.object(torch.cuda, 'is_available', return_value=False):
            result = generator.generate_prompt_candidates(bundle, 'a traveler', 'red coat', 'blur', {'seed_mode': 'fixed', 'llm_seed': 42})
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, 'unbalanced_prompt_syntax')
        self.assertFalse(result.candidate_debug[0]['accepted'])
        self.assertEqual(prompt_builder.build_final_positive(result.selected_positive, 'red coat'), 'a traveler, red coat')

    def test_gguf_fallback_also_rejects_unfinished_weight(self):
        model = types.SimpleNamespace(create_chat_completion=lambda **kwargs: iter([{'choices': [{'delta': {'content': 'portrait, (bravery:1'}}]}]))
        bundle = types.SimpleNamespace(spec=types.SimpleNamespace(backend='llama_cpp'), model=model, load_debug={'backend': 'llama_cpp'})
        result = generator.generate_prompt_candidates(bundle, 'a traveler', 'red coat', 'blur', {})
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.fallback_reason, 'unbalanced_prompt_syntax')
        self.assertFalse(result.candidate_debug[0]['accepted'])
        self.assertEqual(prompt_builder.build_final_positive(result.selected_positive, 'red coat'), 'a traveler, red coat')
if __name__ == '__main__':
    unittest.main()
