import argparse
import torch

from dynrefer.common.config import Config
from dynrefer.rl.grpo import ViewSelector, GRPOTrainer, DynReferEnv
from dynrefer.tasks import *
from dynrefer.datasets import *
from dynrefer.models import *
from dynrefer.runners import *


def parse_args():
    parser = argparse.ArgumentParser(description="Train view selector with GRPO")
    parser.add_argument("--cfg-path", type=str, required=True)
    parser.add_argument("--options", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(args)
    task = DynReferTask.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    env = DynReferEnv(model)

    state_dim = cfg.model_cfg.get("state_dim", 768)
    num_views = cfg.model_cfg.get("num_views", 3)
    selector = ViewSelector(state_dim, 256, num_views, num_select=3)
    trainer = GRPOTrainer(selector, env)

    train_loader = datasets["train"]
    for epoch in range(1):
        for sample in train_loader:
            loss, reward = trainer.train_batch(sample)
            print(f"loss: {loss:.4f}, reward: {reward:.4f}")

    torch.save(selector.state_dict(), "view_selector.pth")


if __name__ == "__main__":
    main()
