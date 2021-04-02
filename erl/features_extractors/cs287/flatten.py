from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

class MyFlattenExtractor(FlattenExtractor):
    """
    control group: simply flatten
    """
    pass