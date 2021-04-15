import torch as th

import erl.features_extractors.cs253 as extractors
from erl.customized_agents.multi_extractor import MultiExtractor
from erl.experiments.cs253.multimodule_exp import MultiModuleExp

def run_current_exp(args):
    th.manual_seed(args.seed)
    run_cs253(args)

def run_cs253(args):
    extractor, extractor_kwargs = extractors.get(args.extractor)
    assert extractor==MultiExtractor
    MultiModuleExp(env_id=args.env_id, features_extractor_kwargs=extractor_kwargs, args=args).train()
