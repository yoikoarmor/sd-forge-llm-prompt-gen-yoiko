import types
import unittest
from unittest.mock import Mock
from backend import generator
CHAT_TEMPLATE = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% if enable_thinking is defined and enable_thinking is false %}{{ '<think>\n\n</think>\n\n' }}{% else %}{{ '<think>\n' }}{% endif %}{% endif %}"

class ChatTemplatePreservationTests(unittest.TestCase):

    def setUp(self):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        self.torch = torch
        tokens = ['[UNK]', '[PAD]', '<|im_start|>', '<|im_end|>', '<think>', '</think>', 'system', 'user', 'assistant', 'portrait', 'soft', 'light']
        backend = Tokenizer(models.WordLevel({token: i for (i, token) in enumerate(tokens)}, unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]', eos_token='<|im_end|>', additional_special_tokens=['<|im_start|>', '<think>', '</think>'], chat_template=CHAT_TEMPLATE)
        self.model = types.SimpleNamespace(parameters=lambda : iter([torch.empty(0)]))
        self.messages = [{'role': 'system', 'content': 'soft light'}, {'role': 'user', 'content': 'portrait'}]

    def test_no_thinking_input_matches_direct_template_tokenization(self):
        rendered = self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        expected_encoding = self.tokenizer.apply_chat_template(self.messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors='pt')
        expected_ids = expected_encoding['input_ids'][0].tolist()
        self.assertTrue(rendered.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'))
        prepared = generator._prepare_inputs(self.tokenizer, self.model, messages=self.messages, enable_thinking=False)
        self.assertEqual(prepared.prompt_text, rendered)
        self.assertEqual(prepared.encoded['input_ids'][0].tolist(), expected_ids)
        self.assertTrue(prepared.chat_template_used)
        self.assertFalse(prepared.empty_think_block_stripped)

    def test_thinking_and_unspecified_modes_also_preserve_template_output(self):
        for enable_thinking in (True, None):
            with self.subTest(enable_thinking=enable_thinking):
                kwargs = {} if enable_thinking is None else {'enable_thinking': enable_thinking}
                rendered = self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True, **kwargs)
                expected_encoding = self.tokenizer.apply_chat_template(self.messages, tokenize=True, add_generation_prompt=True, **kwargs, return_dict=True, return_tensors='pt')
                expected_ids = expected_encoding['input_ids'][0].tolist()
                prepared = generator._prepare_inputs(self.tokenizer, self.model, messages=self.messages, enable_thinking=enable_thinking)
                self.assertEqual(prepared.prompt_text, rendered)
                self.assertEqual(prepared.encoded['input_ids'][0].tolist(), expected_ids)
                self.assertFalse(prepared.empty_think_block_stripped)

    def test_legacy_template_fallback_is_preserved_verbatim(self):
        rendered = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        tokenizer = Mock(wraps=self.tokenizer)
        tokenizer.chat_template = CHAT_TEMPLATE
        tokenizer.apply_chat_template = Mock(side_effect=[TypeError('unsupported enable_thinking'), rendered])
        prepared = generator._prepare_inputs(tokenizer, self.model, messages=self.messages, enable_thinking=False)
        self.assertEqual(prepared.prompt_text, rendered)
        self.assertEqual(tokenizer.call_args.args[0], rendered)
        self.assertFalse(tokenizer.call_args.kwargs['add_special_tokens'])
        self.assertNotIn('enable_thinking', tokenizer.apply_chat_template.call_args.kwargs)

    def test_generation_receives_and_logs_unmodified_no_thinking_input(self):
        self.model.generate = Mock(side_effect=lambda input_ids, **kwargs: self.torch.cat((input_ids, self.torch.tensor([[9, 10, 11]])), dim=1))
        payload = generator._build_input_payload('simple_chat_template', 'portrait', 'coat', 'blur')
        rendered = self.tokenizer.apply_chat_template(payload['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        expected_encoding = self.tokenizer.apply_chat_template(payload['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors='pt')
        expected_ids = expected_encoding['input_ids'][0].tolist()
        result = generator.generate_prompt_candidates(types.SimpleNamespace(model=self.model, tokenizer=self.tokenizer, load_debug={}), 'portrait', 'coat', 'blur', {'enable_thinking': False, 'seed_mode': 'fixed', 'llm_seed': 42})
        actual_ids = self.model.generate.call_args.kwargs['input_ids'][0].tolist()
        self.assertEqual(actual_ids, expected_ids)
        self.assertEqual(result.input_debug['llm_input_text_full'], rendered)
        self.assertFalse(result.input_debug['empty_think_block_stripped'])
if __name__ == '__main__':
    unittest.main()
