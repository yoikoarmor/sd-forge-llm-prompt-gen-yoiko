import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch
from backend import registry, runtime

class RuntimeCudaMemoryTests(unittest.TestCase):

    def fake_torch(self):
        import torch
        events = []

        @contextmanager
        def device(index):
            events.append(('device', index))
            yield

        @contextmanager
        def pool_context(pool):
            events.append('pool entered')
            yield
            events.append('pool exited')
        cuda = types.SimpleNamespace(is_available=Mock(return_value=True), device_count=lambda : 3, current_device=lambda : 0, device=device, MemPool=object, use_mem_pool=pool_context, current_blas_handle=lambda : events.append('blas'))
        return (types.SimpleNamespace(cuda=cuda, device=torch.device), events)

    def test_explicit_devices_get_isolated_workspaces(self):
        (torch, events) = self.fake_torch()
        spec = registry.ModelSpec('test', 'local', device_map={'a': 'cpu', 'b': 1, 'c': 'cuda:1', 'd': 'cuda:2', 'e': 'disk'})
        with patch.dict(sys.modules, {'torch': torch}):
            runtime._prepare_cuda_workspaces(spec)
        self.assertEqual(events, [('device', 1), 'pool entered', 'blas', 'pool exited', ('device', 2), 'pool entered', 'blas', 'pool exited'])

    def test_cpu_and_gguf_do_not_initialize_cuda(self):
        (torch, events) = self.fake_torch()
        for options in [{'device_map': 'cpu'}, {'device_map': None}, {'backend': 'llama_cpp'}]:
            with self.subTest(options=options), patch.dict(sys.modules, {'torch': torch}):
                runtime._prepare_cuda_workspaces(registry.ModelSpec('test', 'local', **options))
        torch.cuda.is_available.assert_not_called()
        self.assertEqual(events, [])

    def test_no_cuda_skips_workspace_preparation(self):
        (torch, events) = self.fake_torch()
        torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {'torch': torch}):
            runtime._prepare_cuda_workspaces(registry.ModelSpec('test', 'local'))
        self.assertEqual(events, [])

    def test_missing_pool_support_does_not_block_loading(self):
        (torch, _) = self.fake_torch()
        del torch.cuda.MemPool
        logger = Mock()
        spec = registry.ModelSpec('test', 'local')
        bundle = types.SimpleNamespace(spec=spec, model=object(), tokenizer=object(), load_debug={})
        rt = runtime.LlmRuntime()
        with patch.dict(sys.modules, {'torch': torch}), patch.object(runtime, 'load_model_bundle', return_value=bundle):
            self.assertEqual(rt.ensure_loaded(spec, logger=logger), (bundle, True))
        self.assertTrue(any(('workspace preparation skipped' in str(call) for call in logger.call_args_list)))

    def test_new_load_and_reuse_both_prepare_current_thread_workspace(self):
        spec = registry.ModelSpec('test', 'local')
        bundle = types.SimpleNamespace(spec=spec, model=object(), tokenizer=object(), load_debug={})
        events = []
        rt = runtime.LlmRuntime()
        with patch.object(runtime, '_prepare_cuda_workspaces', side_effect=lambda *a, **k: events.append('workspace')), patch.object(runtime, 'load_model_bundle', side_effect=lambda *a, **k: events.append('load') or bundle):
            self.assertEqual(rt.ensure_loaded(spec), (bundle, True))
            self.assertEqual(rt.ensure_loaded(spec), (bundle, False))
        self.assertEqual(events, ['workspace', 'load', 'workspace'])
if __name__ == '__main__':
    unittest.main()
