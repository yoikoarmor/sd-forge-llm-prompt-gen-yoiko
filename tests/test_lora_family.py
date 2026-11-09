import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from backend import loader, prompt_builder, registry
from tools import configure_lora_family as configurator

class LoraFamilyTests(unittest.TestCase):

    def test_prompt_is_exact_manifest_and_user_request(self):
        family = json.loads((registry.CONFIG_DIR / 'lora_family.json').read_text())
        self.assertEqual(prompt_builder.SYSTEM_PROMPT, family['system_prompt'])
        self.assertEqual(prompt_builder.build_simple_chat_messages('  departure  ', 'old', 'bad'), [{'role': 'system', 'content': family['system_prompt']}, {'role': 'user', 'content': 'departure'}])
        self.assertEqual(len(family['models']), 7)
        self.assertTrue(all((x['training_epoch'] == 3 for x in family['models'])))

    def test_disabled_choices_are_hidden(self):
        with patch.object(registry, '_get_registry_data', return_value={'models': {'disabled': {'enabled': False}, 'ok': {'enabled': True}}}):
            self.assertEqual(registry.get_ui_model_choices(), ['none', 'ok'])

    def test_missing_windows_path_not_remote(self):
        for module in (registry, loader):
            self.assertFalse(module._looks_like_remote_id('C:\\missing'))
            self.assertFalse(module._looks_like_remote_id('./artifacts'))
            self.assertFalse(module._looks_like_remote_id('/missing/path'))
            self.assertTrue(module._looks_like_remote_id('Qwen/Qwen3.5-9B'))

    def test_pins_are_in_signature(self):
        a = registry.ModelSpec('test', 'Qwen/Qwen3.5-9B', base_model_revision='a')
        b = registry.ModelSpec('test', 'Qwen/Qwen3.5-9B', base_model_revision='b')
        self.assertNotEqual(a.signature(), b.signature())

    def test_snapshot_keeps_revision_and_standard_cache(self):
        download = Mock(return_value='/resolved/snapshot')
        cache = Mock(return_value=None)
        with patch.object(loader, '_import_huggingface_hub', return_value=(download, cache)):
            result = loader._resolve_reference('Qwen/Qwen3.5-9B', label='base', cache_dir=None, local_files_only=True, revision='fixed')
        self.assertEqual(result.resolved_reference, '/resolved/snapshot')
        self.assertEqual(download.call_args.kwargs['revision'], 'fixed')
        self.assertTrue(download.call_args.kwargs['local_files_only'])
        self.assertNotIn('local_dir', download.call_args.kwargs)
        self.assertEqual(cache.call_args.kwargs['revision'], 'fixed')
        self.assertIn('*.jinja', loader.BASE_MODEL_ALLOW_PATTERNS)
        self.assertIn('merges.txt', loader.BASE_MODEL_ALLOW_PATTERNS)

    def test_adapter_hash_rejects_changed_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'adapter_model.safetensors'
            path.write_bytes(b'verified')
            reference = loader.ResolvedReference(tmp, 'local', resolved_local_path=tmp)
            loader._verify_adapter_hash(reference, hashlib.sha256(b'verified').hexdigest())
            with self.assertRaisesRegex(loader.ModelLoadError, 'SHA256 mismatch'):
                loader._verify_adapter_hash(reference, '0' * 64)

    def test_merged_model_loads_without_peft(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = registry.ModelSpec('merged', tmp, load_in_4bit=False, tokenizer_source='base', chat_template_source='base')
            tokenizer = SimpleNamespace(pad_token='pad', eos_token='end', chat_template='template')
            auto_tokenizer = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
            model = SimpleNamespace(eval=Mock())
            auto_model = SimpleNamespace(from_pretrained=Mock(return_value=model))
            torch = SimpleNamespace(float16='fp16', bfloat16='bf16', float32='fp32', cuda=SimpleNamespace(is_available=lambda: False))
            with patch.object(loader, '_import_torch', return_value=torch), patch.object(loader, '_import_transformers', return_value=(auto_model, auto_tokenizer, None)), patch.object(loader, '_import_peft', side_effect=AssertionError('PEFT must not load')):
                bundle = loader.load_model_bundle(spec)
            self.assertIs(bundle.model, model)
            self.assertFalse(bundle.load_debug['adapter_weights_loaded'])
            self.assertEqual(bundle.load_debug['chat_template_source'], 'base')

    def test_configure_remote_merge_preserves_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'registry.json'
            path.write_text(json.dumps({'models': {'custom': {'enabled': True}}}))
            configurator.configure(path, merged_model='test/model', merged_revision='a' * 40)
            data = json.loads(path.read_text())
            self.assertIn('custom', data['models'])
            entry = data['models']['qwen3.5-9b-lora7-merged']
            self.assertIsNone(entry['adapter_path'])
            self.assertEqual(entry['base_model_revision'], 'a' * 40)
            self.assertEqual(entry['tokenizer_source'], 'base')
            self.assertEqual(entry['chat_template_source'], 'base')
            self.assertFalse(entry['local_files_only'])

    def test_unpinned_remote_merge_does_not_modify_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'registry.json'
            path.write_text('{"models":{}}')
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'merged-revision'):
                configurator.configure(path, merged_model='test/model')
            self.assertEqual(path.read_bytes(), before)

    def test_family_registration_is_atomic_on_hash_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry_path = root / 'registry.json'
            registry_path.write_text('{"models":{"custom":{"enabled":true}}}')
            before = registry_path.read_bytes()
            family = json.loads((registry.CONFIG_DIR / 'lora_family.json').read_text())
            for item in family['models']:
                adapter = root / item['version'] / 'best_eval_adapter'
                adapter.mkdir(parents=True)
                for name in ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
                    (adapter / name).write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                configurator.configure(registry_path, adapter_root=root)
            self.assertEqual(registry_path.read_bytes(), before)

    def test_missing_local_merge_cannot_be_treated_as_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'registry.json'
            with self.assertRaisesRegex(ValueError, 'existing directory'):
                configurator.configure(path, merged_model='Z:\\missing', merged_revision='a' * 40)
            self.assertFalse(path.exists())
if __name__ == '__main__':
    unittest.main()
