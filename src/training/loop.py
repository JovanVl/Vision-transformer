import torch

from torch.utils.data import DataLoader, random_split

def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]
        if len(logits.shape) == 1:
            preds = logits > 0
            total_correct += sum(preds == labels).item()
        else:
            _, indices = torch.max(logits, 1)
            total_correct += sum(indices == labels).item()

    return total_loss / total_samples, total_correct / total_samples



@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]
        if len(logits.shape) == 1:
            preds = logits > 0
            total_correct += sum(preds == labels).item()
        else:
            _, indices = torch.max(logits, 1)
            total_correct += sum(indices == labels).item()

    model.train()
    return total_loss / total_samples, total_correct / total_samples



def fit(model, dataset, optimizer, criterion, epochs=1, batch_size=16, validation_split=0.2, device='cpu', checkpoint_path=None):
    validation_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - validation_size

    train_set, validation_set = random_split(dataset, [train_size, validation_size])
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    model.to(device)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, validation_dataloader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if checkpoint_path is not None and best_val_acc < val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)


        print(f'epoch [{epoch + 1}/{epochs}]')
        print(f'train loss: {train_loss:.4f}, train accuracy: {train_acc:.4f}')
        print(f'validation loss: {val_loss:.4f}, validation accuracy: {val_acc:.4f}')

    return history



@torch.no_grad()
def predict(model, dataset, batch_size=16, device='cpu'):
    model.eval()

    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []

    for images, _ in test_dataloader:
        images = images.to(device)
        logits = model(images)
        
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
    
    return torch.cat(all_preds)
