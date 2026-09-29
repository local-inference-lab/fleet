import ast
import sys
import types
from pathlib import Path


class Tensor:
    def __init__(self, shape):
        self.shape = shape

    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


calls = []


def norm(x, weight, bias, eps, **kwargs):
    assert weight.shape == (x.shape[-1],)
    assert 0 < kwargs['group_size'] <= x.shape[-1]
    assert kwargs['out'].shape == x.shape == kwargs['z'].shape
    assert kwargs['norm_before_gate'] is False
    assert kwargs['activation'] == 'silu'
    calls.append(x.shape)


module = types.ModuleType('vllm.third_party.flash_linear_attention.ops.layernorm_guard')
module.layer_norm_fwd = norm
sys.modules[module.__name__] = module
tree = ast.parse(Path(sys.argv[1] if len(sys.argv) > 1 else '/opt/venv/lib/python3.12/site-packages/vllm/model_executor/warmup/qwen_triton_warmup.py').read_text())
method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_warm_layer_norm_kernel')
namespace = {
    'torch': types.SimpleNamespace(device=object, empty=lambda shape, **kwargs: Tensor(shape), empty_like=lambda t: Tensor(t.shape)),
    '_QwenGDNWarmupConfig': types.SimpleNamespace,
}
exec(compile(ast.Module(body=[method], type_ignores=[]), '<Qwen norm warmup test>', 'exec'), namespace)
for width, group_size in ((128, 6144), (6144, 128)):
    calls.clear()
    config = types.SimpleNamespace(hv=48, v=128, norm_weight=Tensor((width,)), norm_group_size=group_size, norm_bias=None, norm_eps=1e-6, norm_before_gate=False, norm_activation='silu', conv_dtype='bf16')
    namespace['_warm_layer_norm_kernel']('test', config)
    assert len(calls) == 6
    for shape, length in zip(calls, (1, 2, 16, 32, 128, 1024)):
        assert shape[0] * shape[1] == length * config.hv * config.v
    print(f'PASS norm width={width}: correct rows and feature width for all warmup lengths')
