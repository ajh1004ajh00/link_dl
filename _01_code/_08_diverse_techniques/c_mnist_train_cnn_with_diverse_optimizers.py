import torch
from torch import optim
from datetime import datetime
import os
import wandb
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")
if not os.path.isdir(CHECKPOINT_FILE_PATH):
  os.makedirs(os.path.join(CURRENT_FILE_PATH, "checkpoints"))

import sys
sys.path.append(BASE_PATH)

from _01_code._06_fcn_best_practice.f_mnist_train_fcn import get_data
from _01_code._07_cnn.a_mnist_train_cnn import get_cnn_model
from _01_code._08_diverse_techniques.a_arg_parser import get_parser
from _01_code._08_diverse_techniques.b_trainer import ClassificationTrainerNoEarlyStopping


def main(args):
  config = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'validation_intervals': args.validation_intervals,
    'print_epochs': args.print_epochs,
    'learning_rate': args.learning_rate,
  }

  optimizer_names = ["SGD", "Momentum", "RMSProp", "Adam"]
  run_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')
  name = "{0}_{1}".format(optimizer_names[args.optimizer], run_time_str)

  wandb.init(
    mode="online" if args.wandb else "disabled",
    project="cnn_mnist_with_diverse_optimizers",
    notes="mnist experiment with cnn and diverse optimizers",
    tags=["cnn", "mnist" "diverse_optimizers"],
    name=name,
    config=config
  )
  print(args)
  print(wandb.config)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Training on device {device}.")

  train_data_loader, validation_data_loader, mnist_transforms = get_data(flatten=False)
  model = get_cnn_model()
  model.to(device)
  wandb.watch(model)

  optimizers = [
    optim.SGD(model.parameters(), lr=wandb.config.learning_rate),
    optim.SGD(model.parameters(), lr=wandb.config.learning_rate, momentum=0.9),
    optim.RMSprop(model.parameters(), lr=wandb.config.learning_rate),
    optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
  ]

  print("Optimizer:", optimizers[args.optimizer])

  classification_trainer = ClassificationTrainerNoEarlyStopping(
    "mnist", model, optimizers[args.optimizer],
    train_data_loader, validation_data_loader, mnist_transforms,
    run_time_str, wandb, device, CHECKPOINT_FILE_PATH
  )
  classification_trainer.train_loop()

  wandb.finish()


if __name__ == "__main__":
  parser = get_parser()
  args = parser.parse_args()
  main(args)
  # python _01_code/_08_diverse_techniques/c_mnist_train_cnn_with_diverse_optimizers.py --wandb -o 2 -v 1
  # python _01_code/_08_diverse_techniques/c_mnist_train_cnn_with_diverse_optimizers.py --no-wandb -o 2 -v 1