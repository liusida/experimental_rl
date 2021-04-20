import numpy as np
import torch as th

from erl.customized_agents.multi_extractor import MultiExtractor
from erl.customized_agents.multi_extractor_with_cnn import MultiExtractorWithCNN
from erl.experiments.cs253.multimodule_exp import MultiModuleExp

def run_current_exp(args):
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    args.group = f"I{int(args.implementation_check)}F{int(args.flatten)}R{args.num_rnns}M{args.num_mlps}"
    if args.implementation_check:
        run_default(args)
    else:
        run_cs253(args)

def run_cs253(args):
    extractor_kwargs = {
        "flatten": args.flatten,
        "num_rnns": args.num_rnns,
        "num_mlps": args.num_mlps,
        "rnn_layer_size": args.rnn_layer_size,
    }

    MultiModuleExp(env_id=args.env_id, features_extractor_class=MultiExtractor if not args.cnn else MultiExtractorWithCNN, features_extractor_kwargs=extractor_kwargs, args=args).train()

def run_default(args):
    """ default flatten version, no customization.
    For comparing to flatten group, to verify the implementation"""
    from erl.experiments.cs253.baseline import BaselineExp
    exp = BaselineExp(env_id=args.env_id, args=args)
    exp.train()