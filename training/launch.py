# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
from train_utils.general import safe_makedirs

import sys
sys.path.append("/home/data/wxz/projects/FFInstrinsic/pi3")

def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config)

    # save config
    safe_makedirs(cfg.logging.log_dir)
    config_save_path = os.path.join(cfg.logging.log_dir, "config.yaml")
    OmegaConf.save(cfg, config_save_path)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()


