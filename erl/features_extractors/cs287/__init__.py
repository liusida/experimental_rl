from .flatten import MyFlattenExtractor
from .one_mlp import OneMlpExtractor
from .two_mlp import TwoMlpExtractor
from .multi_mlp import MultiMlpExtractor

_extractors = {
    "MyFlattenExtractor": MyFlattenExtractor,
    "OneMlpExtractor": OneMlpExtractor,
    "TwoMlpExtractor": TwoMlpExtractor,
    "MultiMlpExtractor": MultiMlpExtractor,
}

def get(classname):
    """ Get class and kwargs from a string of classname
    For example a classname can be: "SomeClass:i=1&j=2"
    and we should return the class of SomeClass, and a dictionary of {"i":1, "j":2}
    
    Note: The value of arguments should always be integers.
    """
    classname, str_kwargs = classname.split(":")
    list_kwargs = str_kwargs.split("&")
    kwargs = {}
    for k in list_kwargs:
        key, value = k.split("=")
        kwargs[key] = int(value)
    return _extractors[classname], kwargs