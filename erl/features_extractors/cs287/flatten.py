from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

class MyFlattenExtractor(FlattenExtractor):
    """
    control group: simply flatten
    """
    # This class is very important, because it helped me figure out where the bug was.
    # Note: net_arch must be specified when we replace the feature extractors
    # because sb3 won't set the default network architecture if we change the features_extractor.
    pass
