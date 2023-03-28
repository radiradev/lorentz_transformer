import torch
import torch.nn as nn
import numpy as np
from networks.lorentz.transformer import LorentzTransformer


class LorentzClassifier(nn.Module):
    def __init__(self, feature_dim, coords_dim, num_classes, n_hidden=32):
        super().__init__()
        self.transformer = LorentzTransformer(
            dim = n_hidden,
            depth = 8,
            heads=4,
            lorentz_scale=True,
            norm_rel_coors=False,
        )
        self.embedding = nn.Linear(feature_dim, n_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(n_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.feature_dim = feature_dim
        self.coords_dim = coords_dim

    def forward(self, points, features, lorentz_vectors, mask):
        features = self.embedding(features.permute(0, 2, 1))
        lorentz_vectors = lorentz_vectors.permute(0, 2, 1)

        mask = mask.squeeze(1)
        x, _ = self.transformer(feats=features, coors=lorentz_vectors, mask=mask)
        x = x.sum(1) 
        
        x = self.classifier(x)
        return x


def _get_model_info(data_config):
    return {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }



def get_model(data_config, **kwargs):
    feature_dim = len(data_config.input_dicts['pf_features'])
    coords_dim = len(data_config.input_dicts['pf_vectors'])
    num_classes = len(data_config.label_value)
    model = LorentzClassifier(feature_dim, coords_dim, num_classes)


    model_info = _get_model_info(data_config)

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
