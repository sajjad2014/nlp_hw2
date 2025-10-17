"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.

This file is from Andrej Karpathy's MinGPT.
https://github.com/karpathy/minGPT
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from utils import CfgNode as CN
import copy


class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset, labels=None, val_dataset=None, val_labels=None):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.labels = labels # added for CSCI 662 - classification labels
        self.val_dataset = val_dataset
        self.val_labels = val_labels
        self.best_val_model = None
        self.best_acc = -1
        
        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # dataset splits
        if self.labels is not None:
            # classification
            dataset = torch.utils.data.TensorDataset(self.train_dataset, self.labels)
        else:
            # language modeling
            # X is all but last token, Y is all but first token
            X = self.train_dataset[:, :-1]
            Y = self.train_dataset[:, 1:]
            dataset = torch.utils.data.TensorDataset(X, Y)
        
        # setup the dataloader
        train_loader = DataLoader(
            dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            if self.labels is not None:
                # classification
                self.logits, self.lm_loss, self.classification_logits, self.classification_loss = model(x, classification_targets=y)
            else:
                # language modeling
                self.logits, self.lm_loss, self.classification_logits, self.classification_loss = model(x, targets=y)


            # backprop and update the parameters
            model.zero_grad(set_to_none=True)

            if self.labels is not None:
                # do classification loss
                loss = self.classification_loss
                self.batch_labels = y
            else:
                # do language modeling loss
                loss = self.lm_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            
            if self.iter_num // self.config.iter_per_epoch != (self.iter_num - 1) // self.config.iter_per_epoch:
                if self.val_dataset is not None:
                    model.eval()
                    preds = []
                    for b_start in range(0, len(self.val_dataset), self.config.batch_size):
                        batch_data = self.val_dataset[b_start:b_start + self.config.batch_size]
                        preds += model.classify(batch_data)
                    preds = torch.tensor(preds)
                    acc = (preds == self.val_labels).float().mean().item()
                    if acc > self.best_acc:
                        self.best_acc = acc
                        self.best_val_model = copy.deepcopy(model.state_dict())
                        print(F"Epoch: {self.iter_num // self.config.iter_per_epoch}, Validation Acc: {acc}")
                        print(F"Best Model Updated")
                    model.train()
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
