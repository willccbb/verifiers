import sys
import types
import importlib.machinery

# Minimal torch stub
torch = types.ModuleType('torch')
torch.__spec__ = importlib.machinery.ModuleSpec('torch', loader=None)
torch._dynamo = types.ModuleType('_dynamo')
torch._dynamo.config = types.SimpleNamespace(suppress_errors=True)
torch.utils = types.ModuleType('utils')
torch.utils.data = types.ModuleType('data')
torch.utils.data.DataLoader = object
torch.utils.data.Sampler = object
torch.utils.checkpoint = types.ModuleType('checkpoint')
torch.utils.checkpoint.checkpoint = lambda func, *a, **k: func(*a, **k)
torch.distributed = types.ModuleType('distributed')
torch.distributed.tensor = types.ModuleType('tensor')
torch.distributions = types.ModuleType('distributions')
torch.distributions.constraints = types.ModuleType('constraints')
torch.Tensor = object
class _DummyCallable:
    def __init__(self, *a, **k):
        pass
    def __call__(self, *a, **k):
        return None

class _DummyNN(types.ModuleType):
    def __getattr__(self, name):
        return _DummyCallable

torch.nn = _DummyNN('nn')
class _DummyFunctional(types.ModuleType):
    def __getattr__(self, name):
        return lambda *a, **k: None

torch.nn.functional = _DummyFunctional('functional')
torch.nn.Module = object
torch.nn.CrossEntropyLoss = _DummyCallable
torch.nn.Identity = _DummyCallable

# Minimal peft stub
peft = types.ModuleType('peft')
peft.__spec__ = importlib.machinery.ModuleSpec('peft', loader=None)
class LoraConfig:
    pass
class PeftConfig:
    pass
def get_peft_model(model, config):
    return model
peft.LoraConfig = LoraConfig
peft.PeftConfig = PeftConfig
peft.get_peft_model = get_peft_model

# Minimal accelerate stub
accelerate = types.ModuleType('accelerate')
accelerate.__spec__ = importlib.machinery.ModuleSpec('accelerate', loader=None)
accelerate.utils = types.ModuleType('utils')
accelerate.utils.broadcast_object_list = lambda *a, **k: None
accelerate.utils.gather_object = lambda *a, **k: None
accelerate.utils.is_peft_model = lambda *a, **k: False

sys.modules.setdefault('torch', torch)
sys.modules.setdefault('torch._dynamo', torch._dynamo)
sys.modules.setdefault('torch.utils', torch.utils)
sys.modules.setdefault('torch.utils.data', torch.utils.data)
sys.modules.setdefault('torch.distributed', torch.distributed)
sys.modules.setdefault('torch.distributed.tensor', torch.distributed.tensor)
sys.modules.setdefault('torch.distributions', torch.distributions)
sys.modules.setdefault('torch.distributions.constraints', torch.distributions.constraints)
sys.modules.setdefault('torch.utils.checkpoint', torch.utils.checkpoint)
sys.modules.setdefault('torch.nn.functional', torch.nn.functional)
sys.modules.setdefault('torch.nn', torch.nn)
sys.modules.setdefault('peft', peft)
sys.modules.setdefault('accelerate', accelerate)
sys.modules.setdefault('accelerate.utils', accelerate.utils)
