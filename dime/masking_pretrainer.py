import torch
import torch.optim as optim
import pytorch_lightning as pl
from dime.utils import generate_uniform_mask

class MaskingPretrainer(pl.LightningModule):
    '''
    Pretrain model with missing features.

    Args:
      model: network for predicting response variable.
      mask_layer: layer for masking unobserved features.
      lr: learning rate.
      loss_fn: loss function for training predictor.
      val_loss_fn: loss function for validation performance.
      factor: factor by which to reduce learning rate on plateau.
      patience: number of epochs to wait before reducing learning rate.
      min_lr: minimum learning rate for scheduler.
      early_stopping_epochs: number of epochs to wait before stopping training.
    '''

    def __init__(self,
                 model,
                 mask_layer,
                 lr,
                 loss_fn,
                 val_loss_fn,
                 factor=0.2,
                 patience=2,
                 min_lr=1e-6,
                 early_stopping_epochs=None):
        super().__init__()

        # Save network modules.
        self.model = model
        self.mask_layer = mask_layer
        self.mask_size = self.mask_layer.mask_size

        # Save optimization hyperparameters.
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
        self.early_stopping_epochs = early_stopping_epochs

        # Save loss functions.
        self.loss_fn = loss_fn
        self.val_loss_fn = val_loss_fn

    def on_fit_start(self):
        self.num_bad_epochs = 0
        self.val_pred_list = []
        self.val_label_list = []

    def training_step(self, batch, batch_idx):
        # Setup for minibatch.
        x, y = batch
        mask = generate_uniform_mask(len(x), self.mask_size).to(x.device)

        # Calculate predictions and loss.
        x_masked = self.mask_layer(x, mask)
        pred = self.model(x_masked)
        return self.loss_fn(pred, y)

    def train_epoch_end(self, outputs):
        # Log loss in progress bar.
        loss = torch.stack(outputs).mean()
        self.log('Loss Train', loss, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        # Setup for minibatch.
        x, y = batch
        mask = generate_uniform_mask(len(x), self.mask_size).to(x.device)

        # Calculate predictions.
        x_masked = self.mask_layer(x, mask)
        pred = self.model(x_masked)
        
        # Save predictions and labels.
        self.val_pred_list.append(pred.cpu())
        self.val_label_list.append(y.cpu())

        # Return loss for logging.
        loss = self.val_loss_fn(pred, y)
        return loss

    def on_validation_epoch_end(self):
        if len(self.val_pred_list) > 0:
            # Concatenate predictions and labels.
            pred = torch.cat(self.val_pred_list)
            y = torch.cat(self.val_label_list)

            # Calculate loss and log to progress bar.
            loss = self.val_loss_fn(pred, y)
            self.log('Loss Val', loss, prog_bar=True, logger=True)

            # Perform manual early stopping. Note that this is called before lr scheduler step.
            sch = self.lr_schedulers()
            if loss < sch.best:
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs > self.early_stopping_epochs:
                # Early stopping.
                self.trainer.should_stop = True
        else:
            print("No validation data available for this epoch.")

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=self.factor, patience=self.patience,
            min_lr=self.min_lr, verbose=True)
        return {
            'optimizer': opt,
            'lr_scheduler': scheduler,
            'monitor': 'Loss Val'
        }
