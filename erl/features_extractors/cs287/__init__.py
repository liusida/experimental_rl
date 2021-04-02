from .flatten import MyFlattenExtractor
from .one_mlp import OneMlpExtractor
from .two_mlp import TwoMlpExtractor

_extractors = {
    "MyFlattenExtractor": MyFlattenExtractor,
    "OneMlpExtractor": OneMlpExtractor,
    "TwoMlpExtractor": TwoMlpExtractor,
}

def get(classname):

    return _extractors[classname]