import torch

from pysomer.data import IsomerDataConfig, IsomerDataset
from pysomer.nn import MLP, MLPConfig


data_config = IsomerDataConfig(
    pth="data/isomer_datasets/C4H10-10000.h5",
)
dataset = IsomerDataset(data_config)

config = MLPConfig(
    input_size=14 * 3,
    hidden_size=[64, 64],
    output_size=2,
    dropout=0.1,
)
model = MLP(config)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params:,}")
print(model)

x, y = dataset[0]
y_pred = model(x)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
for epoch in range(10):
    for x, y in dataloader:
        opt.zero_grad()

        y_pred = model(x)

        loss = torch.nn.functional.cross_entropy(y_pred, y)
        loss.backward()

        acc = (y_pred.argmax(dim=1) == y).float().mean()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()

        # print with scientific notation
        print(f"Loss: {loss.item():.2e}, Acc: {acc.item():.2f}")
