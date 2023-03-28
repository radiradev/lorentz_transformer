import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy


class Net(LightningModule):
    """This method has to be inherited and the model class 
    has to be set"""

    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.model = None
        self.learning_rate = learning_rate
        
        # metrics
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')

        self.save_hyperparameters(ignore=['model'])

    def forward(self, points, features, lorentz_vectors, mask):
        return self.model(points, features, lorentz_vectors, mask)
    
    
    def shared_step(self, batch, batch_idx):
        X, _label, _ = batch
        points, features, lorentz_vectors, mask = [X[k] for k in X.keys()]
        # make sure the labels are in the right shape
        y = _label['_label_'].unsqueeze(1)
        mask = mask.bool()

        y_hat = self.forward(points, features, lorentz_vectors, mask)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float()) 
        proba_y = torch.sigmoid(y_hat)

        return loss, y, proba_y

    def training_step(self, batch, batch_idx):
        loss, y, proba_y = self.shared_step(batch, batch_idx)
        number_of_positives = y.sum()

        # update metrics
        self.train_acc(proba_y, y)

        # compute metrics
        acc = self.train_acc.compute()

        log = {
            'train_loss': loss, 
            'train_acc': self.train_acc, 
        }

        self.log('number_of_positives', number_of_positives.float())
        self.log_dict(log, prog_bar=True)
        return loss
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.train_acc)

    def validation_step(self, batch, batch_idx):
        loss, y, proba_y = self.shared_step(batch, batch_idx)
        
        # update metrics
        self.val_acc(proba_y, y)

        # compute metrics

        log = {
            'val_loss': loss,
            'val_acc': self.val_acc,
        }

        self.log_dict(log, prog_bar=True)
        return loss
    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.val_acc)
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
