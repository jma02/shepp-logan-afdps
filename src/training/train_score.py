# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import wandb
import gc
import os
import time
import torch
import numpy as np

import logging
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from sdes.sde import VPSDE, subVPSDE, VESDE

# Keep the import below to register the models in our runtime
from core.advanced_models import ddpm, ncsnv2, ncsnpp

from training.ema import ExponentialMovingAverage
from training.checkpointing import save_checkpoint, restore_checkpoint
from training.data_utils import get_data_scaler, get_data_inverse_scaler
from training.datasets import get_dataloaders
import training.losses as losses
import training.sampling as sampling
from training.configs.config import get_config
from core.advanced_models.utils import create_model

def train(config=get_config(), workdir="shepp-logan-score"):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Initialize model.
  score_model = create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = torch.optim.Adam(score_model.parameters(), lr=config.training.learning_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  os.makedirs(checkpoint_dir, exist_ok=True)
  os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
  
  # Initialize wandb
  wandb.init(project="shepp-logan-afdps", name="ma-jonathan02-sciml", config={})
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Get data loaders
  train_loader, eval_loader = get_dataloaders(config)
  
  # Get iterators
  train_iter = iter(train_loader)
  eval_iter = iter(eval_loader)
  
  # Create data normalizer and its inverse
  scaler = get_data_scaler(config)
  inverse_scaler = get_data_inverse_scaler(config)

  # Setup SDEs 
  # We will use vpsde - Jonathan
  if config.training.sde.lower() == 'vpsde':
    sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))

  for step in range(initial_step, num_train_steps + 1):
    # Get batch and move to device
    batch = next(train_iter)['image'].to(config.device)
    batch = scaler(batch)  # Apply data scaling
    # Execute one training step
    loss = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
      wandb.log({"training_loss": loss.item()}, step=step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      eval_batch = next(eval_iter)['image'].to(config.device)
      eval_batch = scaler(eval_batch)  # Apply data scaling
      eval_loss = eval_step_fn(state, eval_batch)
      logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
      wandb.log({"eval_loss": eval_loss.item()}, step=step)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # Generate and save samples
      if config.training.snapshot_sampling:
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        sample, n = sampling_fn(score_model)
        ema.restore(score_model.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        os.makedirs(this_sample_dir, exist_ok=True)
        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(sample, nrow, padding=2)
        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        # Save numpy array using torch
        torch.save(sample, os.path.join(this_sample_dir, "sample.pt"))
        
        # Save image grid using torchvision
        sample_path = os.path.join(this_sample_dir, "sample.png")
        save_image(image_grid, sample_path)
        
        # Log sample images to wandb
        if step % config.training.log_img_freq == 0:
          wandb.log({"samples": [wandb.Image(sample_path, caption=f"Step {step}")]}, step=step)

def main(config=get_config(), workdir="shepp-logan-score"):
    train(config, workdir);

if __name__ == "__main__":
    main();