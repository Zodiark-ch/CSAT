import argparse
import os
import random
import sys
from datetime import datetime
from importlib import import_module

import numpy as np
import torch
from fastargs import Param, Section, get_current_config
from fastargs.decorators import param
from fastargs.validation import BoolAsInt, File, Folder, OneOf

sys.path.append("src")

Section("overall", "Overall configs").params(
    model_name=Param(str, required=True, default="locuslab/tofu_ft_llama2-7b", desc="Model name"),
    logger=Param(OneOf(["json", "none"]), default="json", desc="Logger to use"),
    cache_dir=Param(Folder(True), default="/root/autodl-tmp/.cache", desc="Cache directory"),
    seed=Param(int, default=0, desc="Random seed"),
)


Section("unlearn", "Unlearning configs").params(
    unlearn_method=Param(
        OneOf(
            [
                "FT",
                "l1_sparse",
                "GA",
                "GA+FT",
                "origin",
                "CL",
                "RL",
                "KL",
                "CL+FT",
                "GA+KL",
                "CL+KL",
                "NPO+FT"
            ]
        ),
        default="GA+FT",
        desc="Unlearning method",
    ),
    num_epochs=Param(int, default=5, desc="Number of epochs to train"),
    lr=Param(float, default=7e-06, desc="Learning rate"),
    weight_decay=Param(float, default=0.1, desc="Weight decay"),
    gradient_accumulation_steps=Param(
        int, default=4, desc="Gradient accumulation steps"
    ),
    mask_path=Param(str, default="/root/autodl-tmp/wagle_mak/tofu_0.95-001.pt", desc="Path to mask file"),
    task_name=Param(
        OneOf(["toxic", "copyright", "tofu", "wmdp"]),
        default="tofu",
        desc="Task name",
    ),
    sophia=Param(BoolAsInt(), default=False, desc="Whether to use SOPHIA"),
    p=Param(float, default=0.95, desc="p for snip_joint"),
    q=Param(float, default=0.95, desc="q for snip_joint"),
    resume_path=Param(
        Folder(False), default=None, desc="Path to resume model for evaluation"
    ),
    max_steps=Param(int, default=-1, desc="Max steps for training"),
    use_lora=Param(BoolAsInt(), default=False, desc="Whether to use LoRA"),
    mu=Param(float, default=1e-06, desc="hessian approximantion parameter"),
)

Section("unlearn.sophia_params", "SOPHIA configs").enable_if(
    lambda cfg: cfg["unlearn.sophia"]
).params(
    betas_low=Param(float, default=0.9, desc="Betas lower for SOPHIA"),
    betas_high=Param(float, default=0.95, desc="Betas higher for SOPHIA"),
    rho=Param(float, default=0.03, desc="Rho for SOPHIA"),
)
Section("unlearn.NPO+FT", "NPO+FT unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.unlearn_method"] == "NPO+FT"
).params(
    gamma=Param(float, default=0.0, desc="hyperparameters before NPO loss"),
)

Section("unlearn.l1_sparse", "L1 sparse unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.unlearn_method"] == "l1_sparse"
).params(
    alpha=Param(float, default=0.0, desc="L1 regularization parameter"),
)

Section("unlearn.CL+KL", "CL+KL unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.unlearn_method"] == "CL+KL"
).params(
    gamma=Param(float, default=0.0, desc="hyperparameters before CL loss"),
)

Section("unlearn.GA+KL", "GA+KL unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.unlearn_method"] == "GA+KL"
).params(
    gamma=Param(float, default=0.0, desc="hyperparameters before GA loss"),
)

Section("unlearn.CL+FT", "CL+FT unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.unlearn_method"] == "CL+FT"
).params(
    gamma=Param(float, default=1.0, desc="hyperparameters before CL loss"),
)
Section("unlearn.GA+FT", "GA+FT unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.unlearn_method"] == "GA+FT"
).params(
    gamma=Param(float, default=1.0, desc="hyperparameters before GA loss"),
)

Section("unlearn.KL", "KL unlearning configs").enable_if(
    lambda cfg: cfg["unlearn.unlearn_method"] == "KL"
).params(
    gamma=Param(
        float, default=0.0, desc="hyperparameters before KL loss on forget dataset"
    ),
)

Section("dataset", "Dataset configs").params(
    forget_dataset_name=Param(str, default="Tofu_forget10", desc="forget dataset name"),
    retain_dataset_name=Param(str, default="Tofu_retain90", desc="retain dataset name"),
    dataset_seed=Param(int, default=1000, desc="Dataset seed"),
    forget_ratio=Param(float, default=400, desc="Forget ratio"),
    self_retain=Param(BoolAsInt(), default=False, desc="Whether to retain self"),
    batch_size=Param(int, default=1, desc="Batch size"),
)

Section("logger", "General logger configs").params(
    name=Param(
        str,
        default=datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
        desc="Name of this run",
    ),
)

Section("logger.json", "JSON logger").enable_if(
    lambda cfg: cfg["overall.logger"] == "json"
).params(
    root=Param(Folder(True), default="files/logs", desc="Path to log folder"),
)


class Main:
    def __init__(self) -> None:
        self.make_config()
        self.setup_seed()
        self.init_model()
        self.init_logger()
        self.run()

    def make_config(self, quiet=False):
        self.config = get_current_config()
        parser = argparse.ArgumentParser("LLM unlearning")
        self.config.augment_argparse(parser)
        self.config.collect_argparse_args(parser)

        self.config.validate()
        if not quiet:
            self.config.summary()

    @param("overall.seed")
    def setup_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @param("overall.model_name")
    def init_model(self, model_name):
        kwargs = self.config.get_section(f"overall")
        kwargs.update(self.config.get_section(f"unlearn"))
        kwargs.update(self.config.get_section(f"dataset"))
        kwargs.update(self.config.get_section(f"unlearn.{kwargs['unlearn_method']}"))
        if kwargs["sophia"]:
            kwargs.update(self.config.get_section(f"unlearn.sophia_params"))
        kwargs["dataset_names"] = {
            "forget": kwargs["forget_dataset_name"],
            "retain": kwargs["retain_dataset_name"],
        }
        self.model = import_module(f"model.unlearn").get(**kwargs)

    @param("overall.logger")
    def init_logger(self, logger):
        kwargs = self.config.get_section(f"logger")
        kwargs.update(self.config.get_section(f"logger.{logger}"))
        kwargs["config"] = self.config.get_all_config()
        self.logger = import_module(f"loggers.{logger}_").get(**kwargs)

    def run(self):
        self.model.run(self.logger)


if __name__ == "__main__":
    Main()
