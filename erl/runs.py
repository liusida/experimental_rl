import torch as th

from erl.customized_agents.multi_extractor import MultiExtractor
from erl.experiments.cs253.multimodule_exp import MultiModuleExp

def run_current_exp(args):
    th.manual_seed(args.seed)
    run_cs253(args)

def run_cs253(args):
    extractor_kwargs = get_kwargs(args.extractor_kwargs)
    MultiModuleExp(env_id=args.env_id, features_extractor_kwargs=extractor_kwargs, args=args).train()

def get_kwargs(str_kwargs):
    """ Get class and kwargs from a string of classname
    For example a classname can be: "SomeClass:i=1&j=2"
    and we should return the class of SomeClass, and a dictionary of {"i":1, "j":2}
    
    Note: The value of arguments should always be integers.
    """
    list_kwargs = str_kwargs.split(",")
    kwargs = {}
    for k in list_kwargs:
        if k.find("=")==-1:
            continue
        key, value = k.split("=")
        kwargs[key] = int(value)
    return kwargs