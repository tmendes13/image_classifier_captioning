import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from train import train, set_seed, get_device, to_device
from data import image_loaders, channel_stats
from load_data import load_coco_dominant10

class MLP(nn.Module):
    def __init__(self, n_inputs=12288, hidden_sizes=None, n_classes=10, dropout=0.2):
        super().__init__()
        layers = []
        in_features = n_inputs

        if hidden_sizes:
            for h in hidden_sizes:
                layers.append(nn.Linear(in_features, h))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_features = h

        layers.append(nn.Linear(in_features, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))

@torch.no_grad()
def evaluate_acc(model, loader):
    model.eval()  
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

test_accs = []
for seed in [0, 1, 2]:
    set_seed(seed)
    
    model = MLP(hidden_sizes=[256, 256], dropout=0.2)
    
    opt = optim.Adam(model.parameters(), lr=3e-4)
    
    hist = train(model, opt, loss_fn, train_loader,
                 val_score_fn=lambda m: evaluate_acc(m, val_loader), epochs=20)
    
    acc = evaluate_acc(model, test_loader)
    test_accs.append(acc)
    print(f"Seed {seed} | Test Accuracy: {100 * acc:.2f}%")

mean_acc = sum(test_accs) / len(test_accs)
print(f"\nMédia Test Accuracy: {100 * mean_acc:.2f}%")
