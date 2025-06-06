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

# Minimal openai stub
openai = types.ModuleType('openai')
class OpenAI:
    def __init__(self, **kwargs):
        pass
openai.OpenAI = OpenAI
sys.modules.setdefault('openai', openai)

# Minimal transformers stub
transformers = types.ModuleType('transformers')
class PreTrainedTokenizerBase:
    def encode(self, text):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if isinstance(messages, list):
            return ''.join(m['content'] for m in messages)
        return messages

transformers.tokenization_utils_base = types.ModuleType('tokenization_utils_base')
transformers.tokenization_utils_base.PreTrainedTokenizerBase = PreTrainedTokenizerBase
sys.modules.setdefault('transformers', transformers)
sys.modules.setdefault('transformers.tokenization_utils_base', transformers.tokenization_utils_base)

# Minimal datasets stub
datasets = types.ModuleType('datasets')
class Dataset(list):
    def __init__(self, data):
        super().__init__(data)
        if data:
            self.column_names = list(data[0].keys())
        else:
            self.column_names = []

    def map(self, func, num_proc=None):
        return Dataset([func(x) for x in self])

    def select(self, indices):
        return Dataset([self[i] for i in indices])

    def shuffle(self, seed=0):
        return self

    @classmethod
    def from_dict(cls, d):
        rows = [dict(zip(d.keys(), vals)) for vals in zip(*d.values())]
        return cls(rows)

datasets.Dataset = Dataset
datasets.load_dataset = lambda *a, **k: Dataset([])
datasets.concatenate_datasets = lambda ds_list: Dataset([item for ds in ds_list for item in ds])
sys.modules.setdefault('datasets', datasets)

# Minimal tqdm stub
import asyncio
tqdm_mod = types.ModuleType('tqdm')
tqdm_asyncio_mod = types.ModuleType('asyncio')

class _TqdmAsync:
    @classmethod
    async def gather(cls, *args, **kwargs):
        kwargs.pop('total', None)
        kwargs.pop('desc', None)
        return await asyncio.gather(*args, **kwargs)

tqdm_asyncio_mod.tqdm_asyncio = _TqdmAsync
tqdm_asyncio_mod.gather = _TqdmAsync.gather
tqdm_mod.asyncio = tqdm_asyncio_mod
sys.modules.setdefault('tqdm', tqdm_mod)
sys.modules.setdefault('tqdm.asyncio', tqdm_asyncio_mod)
