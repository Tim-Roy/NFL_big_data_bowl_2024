import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import os
import csv
import re
from functools import partial
from numpy import inf, mean
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path, PurePath
import yaml
from function import unbalanced_weights, unbalanced_weight_binary, weighted_L1_loss, weighted_loss
from dataset import nfl_touchdown_data, nfl_tackle_data

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
TD_FREQ_10_FRAME = 0.06462859596393301
TD_FREQ = 0.07915540945206756


class touchdown_model_trainer:
    def __init__(
        self,
        model,
        optimizer,
        epochs,
        touchdown_rate,
        train_data_loader,
        test_data_loader=None,
        sensitivity_weight=0.5,
        out_dir="./td_model_logs",
        model_name=None,
        **kwargs,
    ):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if model_name is None:
            dir_names = [PurePath(f).name for f in Path(out_dir).iterdir() if f.is_dir()]
            v_list = [
                int(re.search("(?<=version_)[0-9]+$", f).group(0))
                for f in dir_names
                if bool(re.search("(?<=version_)[0-9]+$", f))
            ]
            v = min(set(list(range(0, len(v_list) + 1))) - set(v_list))
            self.model_name = f"version_{v}"
        else:
            self.model_name = model_name
        self.model = f"{out_dir}/{self.model_name}"
        Path(self.model).mkdir(parents=True, exist_ok=True)

        self.model = model
        self.weight_fn = partial(
            unbalanced_weight_binary,
            positive_rate=touchdown_rate,
            sensitivity_weight=sensitivity_weight,
        )
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=kwargs["lr"], betas=kwargs["betas"]
            )
        elif optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=kwargs["lr"], betas=kwargs["betas"]
            )
        elif optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=kwargs["lr"], momentum=kwargs["momentum"]
            )
        self.epochs = epochs
        self.sensitivity_weight = sensitivity_weight
        self.touchdown_rate = touchdown_rate
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.tb_writer = SummaryWriter(self.model)
        self.batch_count = 0
        hyperparameters = dict()
        hyperparameters["optimizer"] = optimizer
        hyperparameters["sensitivity_weight"] = sensitivity_weight
        hyperparameters["touchdown_rate"] = touchdown_rate
        for x in kwargs:
            hyperparameters[x] = kwargs[x]
        self.hyperparameters = hyperparameters
        self.validation_score = inf
        self.loss = inf
        self.epoch = 0
        self.checkpoint = None

    def train(self):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (playFeatures, target) in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            playFeatures = playFeatures.to(device)
            target = target.to(device)
            pred_out = self.model(playFeatures)
            pred_weight = self.weight_fn(target)
            loss_fn = nn.BCELoss(weight=pred_weight)
            loss = loss_fn(pred_out, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            tb_x = self.batch_count + batch_idx + 1
            self.tb_writer.add_scalar("Batch Loss/train", loss.item(), tb_x)
        print(f"\rEpoch {self.epoch} loss: {running_loss/(batch_idx + 1)}", end="", flush=True)
        self.tb_writer.add_scalar("Epoch Loss/train", running_loss / (batch_idx + 1), self.epoch)
        self.batch_count = tb_x
        self.loss = loss.item()

    def test(self, autosave: bool = False):
        """
        Test returns same loss metrics as train
        and specificity and sensitivity of the classifier
        and the MAE for the x and y axis of the regressor
        projected to original scale (i.e. yards) for easier interperability
        """
        self.model.eval()
        tp = 0
        sensitivity_denom = 0
        tn = 0
        specificity_denom = 0
        loss_list = []
        regression_loss_list = []
        class_loss_list = []
        tb_x = self.batch_count
        epoch = self.epoch
        tp = 0
        sensitivity_denom = 0
        fn = 0
        specificity_denom = 0
        abs_err_x = 0
        abs_err_y = 0
        n = 0
        dummy = True
        with torch.no_grad():
            for playFeatures, target in self.test_data_loader:
                playFeatures = playFeatures.to(device)
                target = target.to(device)
                pred_out = self.model(playFeatures)
                pred_weight = self.weight_fn(target)
                loss_fn = nn.BCELoss(weight=pred_weight)
                loss = loss_fn(pred_out, target)
                loss_list.append(loss)

                tp += (pred_out * target).sum()
                sensitivity_denom += target.sum()
                fn += (pred_out * (torch.ones_like(target) - target)).sum()
                specificity_denom += target.numel() - target.sum()

            sensitivity = float(tp / sensitivity_denom)
            specificity = float(1 - fn / specificity_denom)
            self.tb_writer.add_scalar(
                "Epoch Loss/validation", torch.mean(torch.stack(tuple(loss_list))), epoch
            )
            self.tb_writer.add_scalar(
                "Batch Loss/validation",
                torch.mean(torch.stack(tuple(loss_list))),
                tb_x,
            )
            self.tb_writer.add_scalar("Epoch Sensitivity/validation", sensitivity, epoch)
            self.tb_writer.add_scalar("Epoch Specificity/validation", specificity, epoch)
            self.score_validation(
                sensitivity,
                specificity,
                save_best=(not autosave),
                autosave=autosave,
            )

    def predict(self, playFeatures):
        """
        Assumes a dataloader, e.g. batches of data are provided
        """
        self.model.eval()
        with torch.no_grad():
            playFeatures = playFeatures.to(device)
            x = self.model(playFeatures)
            return x

    def save_hyperparameters(self):
        with open(f"{self.model}/hyperparam.yaml", "w") as f:
            yaml.dump(self.hyperparameters, f, allow_unicode=True, default_flow_style=False)

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            self.train()
            if self.test_data_loader is not None:
                self.test(autosave=(epoch == self.epochs))
        if self.test_data_loader is None:
            self._save_final_checkpoint()

    def score_validation(self, sensitivity, specificity, save_best=True, autosave=False):
        validation_score = (1 - sensitivity) * self.sensitivity_weight + (1 - specificity) * (
            1 - self.sensitivity_weight
        )
        save_epoch = autosave
        if validation_score < self.validation_score:
            self.validation_score = validation_score
            if save_best:
                save_epoch = True
                if self.checkpoint is not None and os.path.isfile(self.checkpoint):
                    Path.unlink(self.checkpoint)

        if save_epoch:
            self._save_checkpoint(
                validation_score, sensitivity=sensitivity, specificity=specificity
            )

    def save_checkpoint(self):
        self.test(autosave=True)

    def _save_checkpoint(self, score, **kwargs):
        params = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.loss,
            "score": score,
        }
        score = int(round(10**4 * float(score)))
        checkpoint = f"{self.model}/model_e{self.epoch}_s{score}.pt"
        for x in kwargs:
            params[x] = kwargs[x]
        torch.save(
            params,
            checkpoint,
        )
        self.checkpoint = checkpoint

    def _save_final_checkpoint(self):
        params = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.loss,
        }
        checkpoint = f"{self.model}/model.pt"
        torch.save(
            params,
            checkpoint,
        )


class tackle_model_trainer:
    def __init__(
        self,
        model,
        device,
        optimizer,
        epochs,
        class_weight,
        x_loss_weight,
        train_data_loader,
        test_data_loader=None,
        sensitivity_weight=0.5,
        out_dir="./model_logs",
        model_name=None,
        **kwargs,
    ):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if model_name is None:
            dir_names = [PurePath(f).name for f in Path(out_dir).iterdir() if f.is_dir()]
            v_list = [
                int(re.search("(?<=version_)[0-9]+$", f).group(0))
                for f in dir_names
                if bool(re.search("(?<=version_)[0-9]+$", f))
            ]
            v = min(set(list(range(0, len(v_list) + 1))) - set(v_list))
            self.model_name = f"version_{v}"
        else:
            self.model_name = model_name
        self.model = f"{out_dir}/{self.model_name}"
        Path(self.model).mkdir(parents=True, exist_ok=True)

        self.model = model
        self.class_weight = class_weight
        self.x_loss_weight = x_loss_weight
        self.regression_loss = partial(weighted_L1_loss, x_weight=x_loss_weight)
        self.loss_fn = partial(weighted_loss, class_weight=class_weight)
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=kwargs["lr"], betas=kwargs["betas"]
            )
        elif optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=kwargs["lr"], betas=kwargs["betas"]
            )
        elif optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=kwargs["lr"], momentum=kwargs["momentum"]
            )
        self.epochs = epochs
        self.sensitivity_weight = sensitivity_weight
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.tb_writer = SummaryWriter(self.model)
        self.batch_count = 0
        hyperparameters = dict()
        hyperparameters["optimizer"] = optimizer
        hyperparameters["class_weight"] = class_weight
        hyperparameters["x_loss_weight"] = x_loss_weight
        hyperparameters["sensitivity_weight"] = sensitivity_weight
        for x in kwargs:
            hyperparameters[x] = kwargs[x]
        self.hyperparameters = hyperparameters
        self.validation_score = inf
        self.loss = inf
        self.epoch = 0
        self.checkpoint = None

    def train(self):
        self.model.train()
        running_loss_class = 0.0
        running_loss_regression = 0.0
        running_loss = 0.0
        for batch_idx, (tackleFeatures, targetClass, targetRegression) in enumerate(
            self.train_data_loader
        ):
            self.optimizer.zero_grad()
            tackleFeatures = tackleFeatures.to(device)
            targetClass = targetClass.to(device)
            targetRegression = targetRegression.to(device)
            class_out, regression_out = self.model(tackleFeatures)
            class_weight = unbalanced_weights(targetClass, self.sensitivity_weight)
            class_loss = nn.BCELoss(reduction="none")
            class_loss = class_loss(class_out, targetClass)
            class_loss *= class_weight
            class_loss = class_loss.mean()
            # Scale by 1/120 to be consistent scale with classification
            regression_loss = self.regression_loss(regression_out, targetRegression) / 120
            loss = self.loss_fn(class_loss, regression_loss)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            running_loss_class += class_loss
            running_loss_regression += regression_loss
            tb_x = self.batch_count + batch_idx + 1
            self.tb_writer.add_scalar("Batch Loss/train", loss.item(), tb_x)
            self.tb_writer.add_scalar("Batch Classification Loss/train", class_loss, tb_x)
            self.tb_writer.add_scalar("Batch Regression Loss/train", regression_loss, tb_x)
        print(f"\rEpoch {self.epoch} loss: {running_loss/(batch_idx + 1)}", end="", flush=True)
        self.tb_writer.add_scalar("Epoch Loss/train", running_loss / (batch_idx + 1), self.epoch)
        self.tb_writer.add_scalar(
            "Epoch Classification Loss/train", running_loss_class / (batch_idx + 1), self.epoch
        )
        self.tb_writer.add_scalar(
            "Epoch Regression Loss/train", running_loss_regression / (batch_idx + 1), self.epoch
        )
        self.batch_count = tb_x
        self.loss = loss.item()

    def test(self, autosave: bool = False):
        """
        Test returns same loss metrics as train
        and specificity and sensitivity of the classifier
        and the MAE for the x and y axis of the regressor
        projected to original scale (i.e. yards) for easier interperability
        """
        self.model.eval()
        tp = 0
        sensitivity_denom = 0
        tn = 0
        specificity_denom = 0
        loss_list = []
        regression_loss_list = []
        class_loss_list = []
        tb_x = self.batch_count
        epoch = self.epoch
        tp = 0
        sensitivity_denom = 0
        fn = 0
        specificity_denom = 0
        abs_err_x = 0
        abs_err_y = 0
        n = 0
        dummy = True
        with torch.no_grad():
            for tackleFeatures, targetClass, targetRegression in self.test_data_loader:
                tackleFeatures = tackleFeatures.to(device)
                targetClass = targetClass.to(device)
                targetRegression = targetRegression.to(device)
                class_out, regression_out = self.model(tackleFeatures)
                class_loss = nn.functional.binary_cross_entropy(class_out, targetClass)
                regression_loss = self.regression_loss(regression_out, targetRegression) / 120
                loss = self.loss_fn(class_loss, regression_loss)
                loss_list.append(loss)
                regression_loss_list.append(regression_loss)
                class_loss_list.append(class_loss)

                tp += (class_out * targetClass).sum()
                sensitivity_denom += targetClass.sum()
                fn += (class_out * (torch.ones_like(targetClass) - targetClass)).sum()
                specificity_denom += targetClass.numel() - targetClass.sum()
                error = targetRegression.permute(1, 0) - regression_out.permute(1, 0)
                abs_err_x += error[0].abs().sum()
                abs_err_y += error[1].abs().sum()
                n += error.shape[1]

            sensitivity = tp / sensitivity_denom
            specificity = 1 - fn / specificity_denom
            mae_x = abs_err_x / n
            mae_y = abs_err_y / n
            self.tb_writer.add_scalar(
                "Epoch Loss/validation", torch.mean(torch.stack(tuple(loss_list))), epoch
            )
            self.tb_writer.add_scalar(
                "Batch Classification Loss/validation",
                torch.mean(torch.stack(tuple(class_loss_list))),
                tb_x,
            )
            self.tb_writer.add_scalar(
                "Batch Regression Loss/validation",
                torch.mean(torch.stack(tuple(regression_loss_list))),
                tb_x,
            )
            self.tb_writer.add_scalar("Classification Sensitivity/validation", sensitivity, epoch)
            self.tb_writer.add_scalar("Classification Specificity/validation", specificity, epoch)
            self.tb_writer.add_scalar("Regression MAE x/validation", mae_x, epoch)
            self.tb_writer.add_scalar("Regression MAE y/validation", mae_y, epoch)
            self.score_validation(
                sensitivity,
                specificity,
                mae_x,
                mae_y,
                save_best=(not autosave),
                autosave=autosave,
            )

    def predict(self, tackleFeatures):
        """
        Assumes a dataloader, e.g. batches of data are provided
        """
        self.model.eval()
        with torch.no_grad():
            tackleFeatures = tackleFeatures.to(device)
            class_out, regression_out = self.model(tackleFeatures)
            return class_out, regression_out

    def save_hyperparameters(self):
        with open(f"{self.model}/hyperparam.yaml", "w") as f:
            yaml.dump(self.hyperparameters, f, allow_unicode=True, default_flow_style=False)

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            self.train()
            if self.test_data_loader is not None:
                self.test(autosave=(epoch == self.epochs))
        if self.test_data_loader is None:
            self._save_final_checkpoint()

    def score_validation(
        self, sensitivity, specificity, mae_x, mae_y, save_best=True, autosave=False
    ):
        class_score = (1 - sensitivity) + (1 - specificity) * 1.5
        regression_score = mae_x + mae_y
        validation_score = class_score + regression_score / 10
        save_epoch = autosave
        if validation_score < self.validation_score:
            self.validation_score = validation_score
            if save_best:
                save_epoch = True
                if self.checkpoint is not None and os.path.isfile(self.checkpoint):
                    Path.unlink(self.checkpoint)

        if save_epoch:
            self._save_checkpoint(
                validation_score,
                sensitivity=sensitivity,
                specificity=specificity,
                mae_x=mae_x,
                mae_y=mae_y,
            )

    def save_checkpoint(self):
        self.test(autosave=True)

    def _save_checkpoint(self, score, **kwargs):
        params = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.loss,
            "score": score,
        }
        score = int(round(10**4 * float(score)))
        checkpoint = f"{self.model}/model_e{self.epoch}_s{score}.pt"
        for x in kwargs:
            params[x] = kwargs[x]
        torch.save(
            params,
            checkpoint,
        )
        self.checkpoint = checkpoint

    def _save_final_checkpoint(self):
        params = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.loss,
        }
        checkpoint = f"{self.model}/model.pt"
        torch.save(
            params,
            checkpoint,
        )
