from dataclasses import dataclass


@dataclass
class TrainerConfig:
    """Trainer configuration for the Isomer dataset."""

    # Number of epochs
    epochs: int = 100

    # Learning rate
    lr: float = 1e-3

    # Batch size
    batch_size: int = 32

    # Number of workers
    num_workers: int = 4

    # Path to the dataset
    pth: str = "data/isomer.h5"

    # Load the dataset in memory
    in_memory: bool = True

    # Path to the model
    model_pth: str = "model.pt"

    # Path to the log
    log_pth: str = "log.csv"

    # Path to the figure
    fig_pth: str = "fig.png"

    # Path to the config
    cfg_pth: str = "config.yaml"


class Trainer:
    """Trainer class for PyTorch model training."""

    def __init__(
        self,
        cfg: TrainerConfig,
    ):
        self.cfg = cfg
        self.model = None
        self.dataloader = None
        self.opt = None
        self.loss_fn = None
        self.acc_fn = None
        self.log = []

    def load_model(self, model):
        self.model = model
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-2
        )
        self.loss_fn = torch.nn.functional.cross_entropy
        self.acc_fn = lambda y_pred, y: (y_pred.argmax(dim=1) == y).float().mean()

    def load_dataloader(self, dataloader):
        self.dataloader = dataloader

    def train(self):
        for epoch in range(self.cfg.epochs):
            for x, y in self.dataloader:
                self.opt.zero_grad()

                y_pred = self.model(x)

                loss = self.loss_fn(y_pred, y)
                loss.backward()

                acc = self.acc_fn(y_pred, y)

                # clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.opt.step()

                self.log.append(
                    {
                        "epoch": epoch,
                        "loss": loss.item(),
                        "acc": acc.item(),
                    }
                )

    def save(self):
        torch.save(self.model.state_dict(), self.cfg.model_pth)
        import pandas as pd

        pd.DataFrame(self.log).to_csv(self.cfg.log_pth)
        import matplotlib.pyplot as plt

        plt.plot([x["loss"] for x in self.log])
        plt.plot([x["acc"] for x in self.log])
        plt.savefig(self.cfg.fig_pth)
        import omegaconf

        omegaconf.OmegaConf.save(omegaconf.OmegaConf.create(self.cfg), self.cfg.cfg_pth)
