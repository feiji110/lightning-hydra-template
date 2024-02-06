#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   gcpnet_module.py
@Time    :   2023/12/14 12:16:40
@Author  :   Hengda.Gao
@Contact :   ghd@nudt.edu.com
'''
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression import MeanAbsoluteError
from torch_geometric.data import Data

class GCPNetLitModule(LightningModule):
    def __init__( 
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
    ) -> None: 
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net
        
        self.criterion = torch.nn.MSELoss()   # MSELoss、L1Loss

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_mae_best = MinMetric()

    def forward(self, g: Data) -> torch.Tensor:
        return self.net(g)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_mae.reset()
        self.val_mae_best.reset()

    def model_step(
        self, batch: Tuple[Data, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g, y = batch,batch.y
        y_hat = self.forward(g)
        loss = self.criterion(y_hat, y)
        return y_hat, y, loss

    def training_step(
        self, batch: Tuple[Data, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        y_hat, y, loss = self.model_step(batch)
        self.train_loss(loss)
        self.train_mae(y_hat, y)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # loss, preds, targets = self.model_step(batch)
        y_hat, y, loss = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)
        self.val_mae(y_hat, y)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)   

    def on_validation_epoch_end(self) -> None:
        mae = self.val_mae.compute()
        self.val_mae_best(mae)
        self.log("val/mae_best", self.val_mae_best.compute(), sync_dist=True, prog_bar=True)
    
    def test_step(self, batch: Tuple[Data, torch.Tensor], batch_idx: int) -> None:
        y_hat, y, loss = self.model_step(batch)
        self.test_loss(loss)
        self.test_mae(y_hat, y)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage =="fit":
            self.net = torch.compile(self.net)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
if __name__ == "__main__":
    _ = GCPNetLitModule(None, None, None, None)