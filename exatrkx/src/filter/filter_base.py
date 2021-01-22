# System imports
import sys
import os

# 3rd party imports
import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import Dataset

from torch_geometric.data import DataLoader
from torch_cluster import radius_graph

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Local imports
from exatrkx.src.utils_torch import graph_intersection
from exatrkx.src import utils_dir

def load_dataset(input_dir, num):
    if not os.path.exists(input_dir):
        return None
    all_events = os.listdir(input_dir)
    all_events = sorted([os.path.join(input_dir, event) for event in all_events])
    loaded_events = [torch.load(event, map_location=torch.device('cpu')) for event in all_events[:num]]

    return loaded_events

class FilterBase(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        '''
        Initialise the Lightning Module that can scan over different filter training regimes
        '''
        # Assign hyperparameters
        self.hparams = hparams
        self.hparams['input_dir'] = utils_dir.embedding_outdir
        self.hparams['output_dir'] = utils_dir.filtering_outdir

    def setup(self, stage):
        datatypes = ["train", "val", "test"]
        input_dirs = [os.path.join(self.hparams["input_dir"], datatype) for datatype in datatypes]
        self.trainset, self.valset, self.testset = [load_dataset(input_dir, self.hparams["train_split"][i])
                                                    for i, input_dir in enumerate(input_dirs)]

    def train_dataloader(self):
        if len(self.trainset) > 0:
            return DataLoader(self.trainset, batch_size=1, num_workers=1)
        else:
            return None

    def val_dataloader(self):
        if len(self.valset) > 0:
            return DataLoader(self.valset, batch_size=1, num_workers=1)
        else:
            return None

    def test_dataloader(self):
        if len(self.testset):
            return DataLoader(self.testset, batch_size=1, num_workers=1)
        else:
            return None

    def configure_optimizers(self):
        optimizer = [torch.optim.AdamW(self.parameters(), lr=(self.hparams["lr"]), betas=(0.9, 0.999), eps=1e-08, amsgrad=True)]
        scheduler = [
            {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[0], factor=self.hparams["factor"], patience=self.hparams["patience"]),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        ]
#         scheduler = [torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=1, gamma=0.3)]
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):

        emb = (None if (self.hparams["emb_channels"] == 0)
               else batch.embedding)  # Does this work??

        if self.hparams['ratio'] != 0:
            num_true, num_false = batch.y.bool().sum(), (~batch.y.bool()).sum()
            fake_indices = torch.where(~batch.y.bool())[0][torch.randint(num_false, (num_true.item()*self.hparams['ratio'],))]
            true_indices = torch.where(batch.y.bool())[0]
            combined_indices = torch.cat([true_indices, fake_indices])
            # Shuffle indices:
            combined_indices[torch.randperm(len(combined_indices))]
            weight = (torch.tensor(self.hparams["weight"]) if ("weight" in self.hparams) 
                      else torch.tensor(self.hparams['ratio'])) 

        else:
            combined_indices = torch.range(batch.e_radius.shape[1])
            weight = (torch.tensor(self.hparams["weight"]) if ("weight" in self.hparams) 
                      else torch.tensor((~batch.y.bool()).sum() / batch.y.sum())) 

        output = (self(torch.cat([batch.cell_data, batch.x], axis=-1), batch.e_radius[:,combined_indices], emb).squeeze()
                  if ('ci' in self.hparams["regime"])
                  else self(batch.x, batch.e_radius[:,combined_indices], emb).squeeze())

        if ('pid' in self.hparams["regime"]):
            y_pid = batch.pid[batch.e_radius[0,combined_indices]] == batch.pid[batch.e_radius[1,combined_indices]]
            loss = F.binary_cross_entropy_with_logits(output, y_pid.float(), pos_weight = weight)
        else:
            loss = F.binary_cross_entropy_with_logits(output, batch.y[combined_indices], pos_weight = weight)


        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        emb = (None if (self.hparams["emb_channels"] == 0)
               else batch.embedding)  # Does this work??

        subset_ind = torch.randint(batch.e_radius.shape[1], (int(batch.e_radius.shape[1]*self.hparams['val_subset']),))

        output = self(torch.cat([batch.cell_data, batch.x], axis=-1), batch.e_radius[:, subset_ind], emb).squeeze() if ('ci' in self.hparams["regime"]) else self(batch.x, batch.e_radius[:, subset_ind], emb).squeeze()

        val_loss = F.binary_cross_entropy_with_logits(output, batch.y[subset_ind])

        self.log('val_loss', val_loss, prog_bar=True)

        #Edge filter performance
        preds = F.sigmoid(output) > self.hparams["filter_cut"] #Maybe send to CPU??
        edge_positive = preds.sum().float()
        if ('pid' in self.hparams["regime"]):
            y_pid = batch.pid[batch.e_radius[0,subset_ind]] == batch.pid[batch.e_radius[1,subset_ind]]
            edge_true = y_pid.sum()
            edge_true_positive = (y_pid & preds).sum().float()
        else:
            edge_true = batch.y[subset_ind].sum()
            edge_true_positive = (batch.y[subset_ind].bool() & preds).sum().float()


        self.log_dict({
            'val_eff': edge_true_positive/edge_true,
            'val_pur': edge_true_positive/edge_positive}, prog_bar=True)


    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (self.trainer.global_step < self.hparams["warmup"]):
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams["warmup"])
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step()
        optimizer.zero_grad()


class FilterBaseBalanced(FilterBase):

    def __init__(self, hparams):
        super().__init__(hparams)
        '''
        Initialise the Lightning Module that can scan over different filter training regimes
        '''
        
    def training_step(self, batch, batch_idx):

        emb = (None if (self.hparams["emb_channels"] == 0)
               else batch.embedding)  # Does this work??

        with torch.no_grad():
            sections = 8
            cut_list = []
            for j in range(sections):
                subset_ind = torch.chunk(torch.arange(batch.e_radius.shape[1]), sections)[j]
                output = self(torch.cat([batch.cell_data, batch.x], axis=-1), batch.e_radius[:, subset_ind], emb).squeeze() if ('ci' in self.hparams["regime"]) else self(batch.x, batch.e_radius[:, subset_ind], emb).squeeze()
                cut = F.sigmoid(output) > self.hparams["filter_cut"]
                cut_list.append(cut)

            cut_list = torch.cat(cut_list)

            num_true, num_false = batch.y.bool().sum(), (~batch.y.bool()).sum()
            true_indices = torch.where(batch.y.bool())[0]
            hard_negatives = cut_list & ~batch.y.bool()
            hard_indices = torch.where(hard_negatives)[0]
            hard_indices = hard_indices[torch.randperm(len(hard_indices))][:int(len(true_indices)*self.hparams["ratio"]/2)]
            easy_indices = torch.where(~batch.y.bool())[0][torch.randint(num_false, (int(num_true.item()*self.hparams['ratio']/2),))]
            
            combined_indices = torch.cat([true_indices, hard_indices, easy_indices])
            
            # Shuffle indices:
            combined_indices[torch.randperm(len(combined_indices))]
            weight = torch.tensor(self.hparams["weight"])

        output = (self(torch.cat([batch.cell_data, batch.x], axis=-1), batch.e_radius[:,combined_indices], emb).squeeze()
                  if ('ci' in self.hparams["regime"])
                  else self(batch.x, batch.e_radius[:,combined_indices], emb).squeeze())

        if ('pid' in self.hparams["regime"]):
            y_pid = batch.pid[batch.e_radius[0,combined_indices]] == batch.pid[batch.e_radius[1,combined_indices]]
            loss = F.binary_cross_entropy_with_logits(output, y_pid.float(), pos_weight = weight)
        else:
            loss = F.binary_cross_entropy_with_logits(output, batch.y[combined_indices], pos_weight = weight)

        # result = pl.TrainResult(minimize=loss)
        # result.log('train_loss', loss, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.shared_evaluation(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        self.shared_evaluation(batch, batch_idx)
        
        
    def shared_evaluation(self, batch, batch_idx):
        
        '''
        This method is shared between validation steps and test steps
        '''
        
        
        emb = (None if (self.hparams["emb_channels"] == 0)
               else batch.embedding)  # Does this work??

        sections = 8
        score_list = []
        val_loss = torch.tensor(0).float()
        for j in range(sections):
            subset_ind = torch.chunk(torch.arange(batch.e_radius.shape[1]), sections)[j]
            output = self(torch.cat([batch.cell_data, batch.x], axis=-1), batch.e_radius[:, subset_ind], emb).squeeze() if ('ci' in self.hparams["regime"]) else self(batch.x, batch.e_radius[:, subset_ind], emb).squeeze()
            scores = F.sigmoid(output) 
            score_list.append(scores)
            if ('pid' not in self.hparams['regime']):
                val_loss = val_loss + F.binary_cross_entropy_with_logits(output, batch.y[subset_ind])
            else:
                y_pid = batch.pid[batch.e_radius[0, subset_ind]] == batch.pid[batch.e_radius[1, subset_ind]]
                val_loss = val_loss + F.binary_cross_entropy_with_logits(output, y_pid)
            
        score_list = torch.cat(score_list)
        cut_list = score_list > self.hparams["filter_cut"]
        
        # result = pl.EvalResult(checkpoint_on=val_loss)
        self.log("val_loss", val_loss, prog_bar=True)
        # result = pl.TrainResult(minimize=val_loss)
        # result.log('val_loss', val_loss)

        #Edge filter performance
        edge_positive = cut_list.sum().float()
        if ('pid' in self.hparams["regime"]):
            y_pid = batch.pid[batch.e_radius[0]] == batch.pid[batch.e_radius[1]]
            edge_true = y_pid.sum()
            edge_true_positive = (y_pid & cut_list).sum().float()
        else:
            edge_true = batch.y.sum()
            edge_true_positive = (batch.y.bool() & cut_list).sum().float()


        self.log_dict({
                'eff': torch.tensor(edge_true_positive/edge_true),
                'pur': torch.tensor(edge_true_positive/edge_positive)})