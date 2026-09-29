import ast
import gc
import sys
import types
import weakref
from pathlib import Path


class Tensor:
    def __setitem__(self, index, value):
        pass

    def zero_(self):
        return self


references = []


def tensor(*args, **kwargs):
    result = Tensor()
    references.append(weakref.ref(result))
    return result


class PreparedCall:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


preparation = types.ModuleType('b12x.preparation')
preparation.PreparedCall = PreparedCall
sys.modules['b12x.preparation'] = preparation
tree = ast.parse(Path(sys.argv[1] if len(sys.argv) > 1 else '/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/b12x.py').read_text())
method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_prepared_call')
namespace = {
    'torch': types.SimpleNamespace(empty=tensor, zeros=tensor, full=tensor, arange=tensor, int32='int32'),
    '_kv_page_size': lambda k, v: 64,
    'PreparationResourceUnavailableError': RuntimeError,
}
exec(compile(ast.Module(body=[method], type_ignores=[]), '<b12x lifetime test>', 'exec'), namespace)
backend = types.SimpleNamespace(
    device='test', num_heads=8, head_size=128, output_head_size=128,
    dtype='bf16', _max_page_table_widths={64: 4096}, window_left=-1, sinks=None,
    _prepare_fp8_descales=lambda *args: (None, None),
)
state = types.SimpleNamespace(
    scratch_plan=types.SimpleNamespace(scratch_specs=lambda: [types.SimpleNamespace(shape=(32,), dtype='bf16')]),
    bind=lambda **kwargs: kwargs,
    run=lambda binding: None,
)
for benchmark in (False, True):
    references.clear()
    call = namespace['_prepared_call'](backend, object(), state, ('extend', 64, 2, 128), benchmark=benchmark, caches=(object(), object()))
    call.produce()
    call.run()
    published_owners = call.owners
    assert bool(published_owners) == benchmark
    del call
    gc.collect()
    assert all((ref() is not None) == benchmark for ref in references)
    del published_owners
    gc.collect()
    assert all(ref() is None for ref in references)
    print(f'PASS benchmark={benchmark}: synthetic buffers have the intended lifetime')
