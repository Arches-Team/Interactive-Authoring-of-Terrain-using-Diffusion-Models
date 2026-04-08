
from functools import wraps
import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import re
import shutil

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from tqdm.auto import tqdm, trange

from .ema import EMA
from .metrics import Metric
from ..core.utils import pad_or_crop_tensor
from ..core.shared import ArgsKwargsWrapper

# Default file names
MODEL_FILE_NAME = 'model.pt'
OPTIMIZER_FILE_NAME = 'optimizer.pt'
SCHEDULER_FILE_NAME = 'scheduler.pt'
EMA_FILE_NAME = 'ema.pt'
SCALER_FILE_NAME = 'scaler.pt'

TRAINER_STATE_FILE_NAME = 'trainer_state.json'

CHECKPOINT_PREFIX = 'checkpoint-'
CHECKPOINT_REGEX = fr'{CHECKPOINT_PREFIX}(\d+)'


def run_if(predicate):
    # https://stackoverflow.com/a/16358316/13989043
    def wrapper(f):
        @wraps(f)
        def wrapped(self, *f_args, **f_kwargs):
            if predicate(self):
                f(self, *f_args, **f_kwargs)
        wrapped.is_callback = True
        return wrapped

    return wrapper


def cycle(it):
    while True:
        yield from it


def regex_search(text, pattern, group=1, default=None):
    match = re.search(pattern, text)
    return match.group(group) if match else default


def list_checkpoints(model_dir):
    checkpoints = {}
    for f in os.listdir(model_dir):
        checkpoint_dir = os.path.join(model_dir, f)
        if not os.path.isdir(checkpoint_dir):
            continue

        match = regex_search(f, CHECKPOINT_REGEX)
        if match is None:
            continue  # No match, ignore

        l = int(match)
        checkpoints[l] = checkpoint_dir

    return checkpoints


def get_latest_checkpoint(model_dir):
    if not os.path.exists(model_dir):
        return None

    checkpoints = list_checkpoints(model_dir)

    if not checkpoints:
        return None

    return checkpoints[max(checkpoints)]


@dataclass
class TrainingArguments:
    num_workers: int = field(
        default=1,
        metadata={
            'help': 'How many subprocesses to use for data loading'
        }
    )

    lr: float = field(
        default=2e-4,
        metadata={
            'help': 'Learning rate'
        }
    )

    num_epochs: int = field(
        default=20,
        metadata={
            'help': 'Number of training epochs'
        }
    )
    save_steps: int = field(
        default=1000,
        metadata={
            'help': 'Save checkpoint every X steps'
        }
    )
    save_limit: Optional[int] = field(
        default=None,
        metadata={
            'help': 'Maximum number of checkpoints to store at a given time (None = no limit)'
        }
    )
    log_steps: int = field(
        default=1000,
        metadata={
            'help': 'Log every X steps'
        }
    )
    eval_steps: int = field(
        default=1000,
        metadata={
            'help': 'Run an evaluation every X steps'
        }
    )

    seed: Optional[int] = field(
        default=None,
        metadata={
            'help': 'Set a seed'
        }
    )

    batch_size: int = field(
        default=1,
        metadata={
            'help': 'The batch size'
        }
    )

    results_dir: str = field(
        default='results_dir',
        metadata={
            'help': 'Where to save results'
        }
    )

    checkpoint: str = field(
        default='latest',
        metadata={
            'help': 'Load model weights and information from this checkpoint'
        }
    )

    amp: bool = field(
        default=False,
        metadata={
            'help': 'Whether to train using automatic mixed precision'
        }
    )

    # Additional training arguments
    use_ema: bool = field(
        default=False,
        metadata={
            'help': 'Whether to train using an exponential moving average model'
        }
    )
    ema_update_every: int = field(
        default=10,
        metadata={
            'help': 'Update EMA model (if specified) every X steps'
        }
    )
    ema_decay: float = field(
        default=0.995,
        metadata={
            'help': 'EMA decay rate'
        }
    )

    # Learning rate scheduler arguments
    lr_scheduler_factor: float = field(
        default=0.8,
        metadata={
            'help': 'Factor by which the learning rate will be reduced'
        }
    )
    lr_scheduler_patience: int = field(
        default=10,
        metadata={
            'help': 'Number of epochs with no improvement after which learning rate will be reduced'
        }
    )
    lr_scheduler_threshold: float = field(
        default=1e-4,
        metadata={
            'help': 'Threshold for measuring the new optimum, to only focus on significant changes'
        }
    )
    lr_scheduler_min_lr: float = field(
        default=1e-6,
        metadata={
            'help': 'A lower bound on the learning rate'
        }
    )
    lr_scheduler_verbose: bool = field(
        default=True,
        metadata={
            'help': 'Whether to print out when learning rate changes'
        }
    )


class ModelInputs(ArgsKwargsWrapper):
    pass


class ModelTargets:
    def __init__(self, targets) -> None:
        self.targets = targets


class BaseTrainer:

    def __init__(self,
                 model: nn.Module,
                 datasets: Dict[str, Dataset],
                 optimizer: optim.Optimizer,
                 criterion,

                 scheduler=None,

                 metrics=None,
                 #  transform_input=None,
                 #  transform_targets=None,
                 transform_output=None,
                 training_args: Optional[TrainingArguments] = None
                 ) -> None:

        self.model = model

        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        # self.transform_inputs = transform_input
        # self.transform_targets = transform_targets
        self.transform_outputs = transform_output

        self.metrics = []
        if metrics is not None:
            for met in metrics:
                if isinstance(met, Metric):
                    self.metrics.append(met)
                elif isinstance(met, type) and issubclass(met, Metric):
                    # Instantiate object of class
                    self.metrics.append(met())
                else:
                    raise ValueError(
                        f'Metric ({met}) is not an instance or subclass of the Metric class')

        # Use default training arguments if not specified
        if training_args is None:
            training_args = TrainingArguments()
        self.training_args = training_args

        self.ema_model = None
        if training_args.use_ema:
            self.ema_model = EMA(self.model,
                                 beta=training_args.ema_decay,
                                 update_every=training_args.ema_update_every)

        self.scaler = GradScaler(enabled=training_args.amp)

        # Set datasets
        if not isinstance(datasets, dict):
            self.datasets = {'train': datasets}
        self.datasets = datasets

        self.dataloaders = {}
        for key, dataset in self.datasets.items():
            self.dataloaders[key] = DataLoader(
                dataset,
                batch_size=self.training_args.batch_size,
                shuffle=key == 'train',
                num_workers=self.training_args.num_workers,
                persistent_workers=True
            )

        checkpoint = training_args.checkpoint
        # For training, or if no model found in model_dir
        if checkpoint == 'latest':  # Infer checkpoint
            checkpoint = get_latest_checkpoint(self.training_args.results_dir)

        self.load_checkpoint(checkpoint)

        self.total_train_steps = self.training_args.num_epochs * \
            len(self.dataloaders['train'])
        self._register_callbacks()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _register_callbacks(self):
        self._callbacks = []
        for name in dir(self):
            item = getattr(self, name)
            if callable(item) and hasattr(item, '__self__') and hasattr(item, 'is_callback') and item.is_callback:
                self._callbacks.append(item)

    def _run_callbacks(self):
        for cb in self._callbacks:
            cb()

    def try_load_state_dict(self, path, obj, try_transfer_learning=True):
        if not os.path.exists(path) or obj is None:
            return

        print(' - loading', path)

        # Load model parameters
        data = torch.load(path, map_location=self.device)
        if isinstance(data, nn.Module):
            data = data.state_dict()

        self._load_state_dict(obj, data, try_transfer_learning)

    def _load_state_dict(self, obj, state_dict, try_transfer_learning):
        if isinstance(obj, nn.Module):
            if try_transfer_learning:
                # Ensure parameters to load are of the correct shape,
                # cropping/padding if necessary (used for transfer learning)
                for name, param in obj.named_parameters():
                    if name in state_dict:  # Try to get weights from model_data
                        # Data exists, so we pad/crop it in each dimension (where necessary)
                        for dim, num in enumerate(param.shape):
                            state_dict[name] = pad_or_crop_tensor(
                                tensor=state_dict[name],
                                dimension=dim,
                                desired_num_values=num,
                            )
                    else:
                        # Unable to find corresponding weights in model_data,
                        # so we randomly initialise. TODO use better technique
                        state_dict[name] = torch.randn(param.shape)

            obj.load_state_dict(state_dict, strict=not try_transfer_learning)

        elif isinstance(obj, optim.Optimizer):
            try:
                obj.load_state_dict(state_dict)
            except ValueError:
                if try_transfer_learning:
                    print('WARNING | Ignoring optimizer due to parameter mismatch')
                else:
                    raise
        else:
            obj.load_state_dict(state_dict)

    def load_checkpoint(self, checkpoint):

        trainer_data = None
        if checkpoint is not None:  # Checkpoint specified, or found above
            print('Loading checkpoint:', checkpoint)

            if os.path.isdir(checkpoint):
                model_path = os.path.join(checkpoint, MODEL_FILE_NAME)

                for file, module in (
                    (OPTIMIZER_FILE_NAME, self.optimizer),
                    (SCHEDULER_FILE_NAME, self.scheduler),
                    (EMA_FILE_NAME, self.ema_model),
                    (SCALER_FILE_NAME, self.scaler),
                ):
                    self.try_load_state_dict(
                        os.path.join(checkpoint, file), module)

            else:
                model_path = checkpoint

            self.try_load_state_dict(model_path, self.model)

            # Try to load trainer state
            trainer_state = os.path.join(checkpoint, TRAINER_STATE_FILE_NAME)
            if os.path.exists(trainer_state):
                with open(trainer_state) as fp:
                    trainer_data = json.load(fp)

        if trainer_data is not None:
            self.steps = trainer_data['steps']
            self.epoch = trainer_data['epoch']
            self.history = trainer_data['history']

        else:  # No trainer_data, set to defaults
            self.steps = 0
            self.epoch = 0
            self.history = []

    def save(self):
        if self.training_args.save_limit is not None and os.path.exists(self.training_args.results_dir):
            # Remove if saving this model will exceed save_limit
            checkpoints = list_checkpoints(self.training_args.results_dir)
            if len(checkpoints) >= self.training_args.save_limit:
                for k in sorted(checkpoints)[:-self.training_args.save_limit+1]:
                    shutil.rmtree(checkpoints[k])

        checkpoint_dir = os.path.join(
            self.training_args.results_dir, f'{CHECKPOINT_PREFIX}{self.steps}')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model separately
        # Note: We save the full model for easier inference later on (stores creation parameters)
        model_path = os.path.join(checkpoint_dir, MODEL_FILE_NAME)
        torch.save(self.model, model_path)

        # Save optimizer
        # https://stackoverflow.com/questions/70768868/pytorch-whats-the-purpose-of-saving-the-optimizer-state
        optimizer_path = os.path.join(checkpoint_dir, OPTIMIZER_FILE_NAME)
        torch.save(self.optimizer.state_dict(), optimizer_path)

        # Save scheduler
        if self.scheduler is not None:
            scheduler_path = os.path.join(checkpoint_dir, SCHEDULER_FILE_NAME)
            torch.save(self.scheduler.state_dict(), scheduler_path)

        # Save EMA
        if self.ema_model is not None:
            ema_model_path = os.path.join(checkpoint_dir, EMA_FILE_NAME)
            torch.save(self.ema_model.state_dict(), ema_model_path)

        # Save Scaler
        if self.scaler is not None:
            scaler_path = os.path.join(checkpoint_dir, SCALER_FILE_NAME)
            torch.save(self.scaler.state_dict(), scaler_path)

        # Save trainer state
        trainer_state = os.path.join(checkpoint_dir, TRAINER_STATE_FILE_NAME)
        with open(trainer_state, 'w') as fp:
            json.dump({
                'steps': self.steps,
                'epoch': self.epoch,
                'history': self.history,
                # 'lr': self.train_lr,
                # 'total_num_epochs': self.num_epochs
            }, fp)

        return checkpoint_dir

    @property
    def current_lr(self):
        # https://discuss.pytorch.org/t/get-current-lr-of-optimizer-with-adaptive-lr/24851/3
        if self.scheduler is not None and hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()

        for param_group in self.optimizer.param_groups:
            return param_group['lr']

        return 0  # No learning rate found (should never reach here)

    def add_to_history(self, type, **kwargs):
        self.history.append({
            'type': type,
            'epoch': self.epoch,
            'steps': self.steps,
            'lr': self.current_lr,
            **kwargs
        })

    def extract(self, item):
        raise NotImplementedError

    def extract_next(self, dataloader) -> Tuple[ModelInputs, ModelTargets]:
        model_input, target = self.extract(next(dataloader))
        if not isinstance(model_input, ModelInputs):
            model_input = ModelInputs(model_input)
        if not isinstance(target, ModelTargets):
            target = ModelTargets(target)

        # TODO move both to correct device

        return model_input, target

    def step(self, dataloader, phase='train'):
        update_gradients = phase == 'train'
        update_metrics = phase != 'train'

        if update_gradients:
            # Zero gradients for every batch
            self.optimizer.zero_grad()

        with torch.set_grad_enabled(update_gradients):

            model_inputs, targets = self.extract_next(dataloader)
            print(model_inputs)

            # https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training
            # Runs the forward pass with autocasting.
            with autocast(enabled=self.training_args.amp):
                outputs = self.model(
                    *model_inputs.args, **model_inputs.kwargs)

                if self.transform_outputs is not None:
                    outputs = self.transform_outputs(outputs)

                loss = self.criterion(outputs, targets.targets)

            if update_metrics:
                for metric in self.metrics:
                    # Add outputs and labels to each metric
                    metric.add(
                        outputs=outputs,
                        targets=targets.targets
                    )

            if update_gradients:
                # Scales loss. Calls backward() on scaled loss to create scaled gradients.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                self.scaler.scale(loss).backward()

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                self.scaler.step(self.optimizer)

                # Updates the scale for next iteration.
                self.scaler.update()

                if self.ema_model is not None:
                    self.ema_model.update()

        return loss.item()

    def train(self):
        train_dataloader = self.dataloaders['train']

        if self.steps >= self.total_train_steps:
            return

        do_eval = 'valid' in self.dataloaders
        do_test = 'test' in self.dataloaders

        loss_steps = 0
        running_loss = 0

        dl = cycle(train_dataloader)
        with tqdm(initial=self.steps, total=self.total_train_steps) as progress:

            while self.steps < self.total_train_steps:
                self.model.train()  # Set model to training mode

                running_loss += self.step(dl)
                loss_steps += 1

                avg_loss = running_loss / loss_steps
                progress.set_description(
                    f'Training | epoch={self.epoch:.2f}, loss={avg_loss:.5f}')

                # Update progress
                progress.update()
                self.epoch = self.steps / len(train_dataloader)
                self.steps += 1

                last_step = self.steps == self.total_train_steps

                # TODO register callbacks:
                if last_step or self.steps % self.training_args.log_steps == 0:
                    self.add_to_history('log', loss=avg_loss)

                    # Reset loss
                    running_loss = 0
                    loss_steps = 0

                if do_eval and (last_step or self.steps % self.training_args.eval_steps == 0):
                    eval_metrics = self.eval()

                    self.add_to_history('eval', **eval_metrics)

                    # Reduce learning rate if no improvement in eval loss
                    if self.scheduler is not None:
                        self.scheduler.step(eval_metrics['loss'])

                if last_step or self.steps % self.training_args.save_steps == 0:
                    progress.set_description('Saving model')
                    if do_test and last_step:
                        # Run testing just before saving final model
                        test_metrics = self.eval(mode='test')
                        self.add_to_history('test', **test_metrics)
                    self.save()

                self._run_callbacks()

    def eval(self, mode='valid'):
        self.model.eval()

        dl = iter(self.dataloaders[mode])
        running_loss = 0
        info_header = 'Testing' if mode == 'test' else 'Validation'
        eval_info = f'{info_header} | epoch={self.epoch:.2f}, steps={self.steps}'

        # Reset metrics
        for metric in self.metrics:
            metric.reset()

        with trange(len(dl)) as progress, torch.no_grad():
            for i in range(len(dl)):
                running_loss += self.step(dl, phase='eval')

                eval_metrics = {
                    'loss': running_loss/(i+1),
                }

                if i == len(dl) - 1:
                    # Only add these metrics at the end
                    for metric in self.metrics:
                        eval_metrics.update(metric.total())

                progress.set_description(
                    f'{eval_info}, {self.format_metrics(eval_metrics)}')
                progress.update()

        return eval_metrics

    @staticmethod
    def format_metrics(metrics_dict: Dict):
        """Helper method to format metrics dictionary"""
        items = [
            f'{k}={v:.5f}'
            for k, v in metrics_dict.items()
        ]
        return ', '.join(items)
