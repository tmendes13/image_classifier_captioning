import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from train import train, set_seed, get_device, to_device
from data import image_loaders, channel_stats
from load_data import load_coco_dominant10

class LinearClassifier(nn.Module):
    def __init__(self, n_inputs=12288, n_classes=10):
        super().__init__()
        self.layer = nn.Linear(n_inputs, n_classes)

    def forward(self, x):
        return self.layer(x.flatten(1))

@torch.no_grad()
def evaluate_acc(model, loader):
    device = next(model.parameters()).device
    correct = total = 0
    for batch in loader:
        x, y = to_device(batch, device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total

def loss_fn(model, batch):
    x, y = batch
    return F.cross_entropy(model(x), y)

data = load_coco_dominant10()
mean, std = channel_stats(data["train"]["images"])
train_loader, val_loader, test_loader = image_loaders(data, batch_size=64, mean=mean, std=std)

for seed in [0, 1, 2]:
    set_seed(seed)
    model = LinearClassifier()
    opt = optim.SGD(model.parameters(), lr=0.01)
    hist = train(model, opt, loss_fn, train_loader,
                 val_score_fn=lambda m: evaluate_acc(m, val_loader), epochs=20)
    test_acc = evaluate_acc(model, test_loader)