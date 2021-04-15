import torch as th

from erl.customized_agents.multi_extractor import MultiExtractor
from erl.experiments.cs253.multimodule_exp import MultiModuleExp

def run_current_exp(args):
    th.manual_seed(args.seed)
    # run_cs253(args)
    run_default(args)

def run_cs253(args):
    extractor_kwargs = {
        "flatten": args.flatten,
        "num_rnns": args.num_rnns,
        "num_mlps": args.num_mlps,
    }
    MultiModuleExp(env_id=args.env_id, features_extractor_kwargs=extractor_kwargs, args=args).train()

def run_default(args):
    """ default flatten version, no customization.
    For comparing to flatten group, to verify the implementation"""
    from erl.experiments.cs253.baseline import BaselineExp
    from stable_baselines3.common.torch_layers import FlattenExtractor
    exp = BaselineExp(env_id=args.env_id, features_extractor_class=FlattenExtractor, args=args)
    exp.train()