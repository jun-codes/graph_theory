import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU

train_set = torch.load(r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project\train.pt", weights_only=False)
val_set   = torch.load(r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project\val.pt", weights_only=False)
test_set  = torch.load(r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project\test.pt", weights_only=False)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)

class GINClassifier(torch.nn.Module):
    def __init__(self, in_channels=5, hidden=64, num_classes=2):
        super().__init__()

        def mlp(in_c, out_c):
            return Sequential(
                Linear(in_c, out_c),
                BatchNorm1d(out_c),
                ReLU(),
                Linear(out_c, out_c),
                ReLU()
            )

        self.conv1 = GINConv(mlp(in_channels, hidden))
        self.conv2 = GINConv(mlp(hidden, hidden))
        self.conv3 = GINConv(mlp(hidden, hidden))

        self.classifier = Sequential(
            Linear(hidden * 2, hidden),
            ReLU(),
            torch.nn.Dropout(0.3),
            Linear(hidden, num_classes)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)

        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=1)

        return self.classifier(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model     = GINClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

def train(loader):
    model.train()
    total_loss, correct = 0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        correct    += out.argmax(dim=1).eq(batch.y.squeeze()).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            batch   = batch.to(device)
            out     = model(batch.x, batch.edge_index, batch.batch)
            correct += out.argmax(dim=1).eq(batch.y.squeeze()).sum().item()
    return correct / len(loader.dataset)

best_val_acc = 0
epochs = 60

print(f"\nTraining for {epochs} epochs...\n")
for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(train_loader)
    val_acc               = evaluate(val_loader)
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),
                   r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project\best_model.pt")

    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
              f"Best Val: {best_val_acc:.4f}")

model.load_state_dict(torch.load(
    r"C:\Users\Arjun\Desktop\code\Graph_Theory_Project\best_model.pt", weights_only=False))
test_acc = evaluate(test_loader)

print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"Best Val Accuracy:   {best_val_acc:.4f}")

