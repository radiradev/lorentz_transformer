from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger

from net import Net
from networks.pfn import ParticleFlowNetwork
from networks.lorentz.lorentz_net import LorentzClassifier 
from dataloader import WeaverDataModule
import argparse

def namespace_to_dict(namespace_obj):
    result = {}
    for key, value in vars(namespace_obj).items():
        if isinstance(value, argparse.Namespace):
            result[key] = namespace_to_dict(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [namespace_to_dict(x) if isinstance(x, argparse.Namespace) else x for x in value]
        else:
            result[key] = value
    return result

class WandbCLI(LightningCLI):
    # TODO should be able to pass in a callback to save the config
    def before_fit(self):
        wandb_config = namespace_to_dict(self.config)
        self.trainer.logger.experiment.config.update(wandb_config)

# set up logger
logger = WandbLogger(project='equivariance', log_model='all', save_code=True)


# TODO this should be in the config file
num_ins = 7
num_outs = 1
learning_rate = 1.0e-05

# TODO these can be in a separate file
class LorentzPL(Net):
    def __init__(self, learning_rate=learning_rate):
        super().__init__(learning_rate)
        self.model = LorentzClassifier(
            num_ins, 
            coords_dim=4, 
            num_classes=num_outs
        )

class PFNPL(Net):
    def __init__(self, learning_rate=learning_rate):
        super().__init__(learning_rate)
        self.model = ParticleFlowNetwork(num_ins, num_outs)
 
# set up the CLI
cli = WandbCLI(datamodule_class=WeaverDataModule, trainer_defaults=dict(logger=logger), save_config_callback=None)
