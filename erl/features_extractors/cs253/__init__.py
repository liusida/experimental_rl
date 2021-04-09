from stable_baselines3.common.torch_layers import FlattenExtractor
from .flatten import MyFlattenExtractor
from .one_mlp import OneMlpExtractor
from .two_mlp import TwoMlpExtractor
from .multi_mlp import MultiMlpExtractor
from .multi_lstm import MultiLSTMExtractor

_extractors = {
    "FlattenExtractor": FlattenExtractor,
    "MyFlattenExtractor": MyFlattenExtractor,
    "OneMlpExtractor": OneMlpExtractor,
    "TwoMlpExtractor": TwoMlpExtractor,
    "MultiMlpExtractor": MultiMlpExtractor,
    "MultiLSTMExtractor": MultiLSTMExtractor,
}

def get(classname):
    """ Get class and kwargs from a string of classname
    For example a classname can be: "SomeClass:i=1&j=2"
    and we should return the class of SomeClass, and a dictionary of {"i":1, "j":2}
    
    Note: The value of arguments should always be integers.
    """
    if classname.find(":")==-1:
        str_kwargs = ""
    else:
        classname, str_kwargs = classname.split(":")
    list_kwargs = str_kwargs.split("&")
    kwargs = {}
    for k in list_kwargs:
        if k.find("=")==-1:
            continue
        key, value = k.split("=")
        kwargs[key] = int(value)
    return _extractors[classname], kwargs