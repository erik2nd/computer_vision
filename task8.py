# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 10, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(10, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(16 * 2 * 2, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (data_batch, target_batch) in enumerate(train_loader):
            data_batch, target_batch = data_batch.to(device), target_batch.to(device)
            data_batch = data_batch.view(-1, 1, 8, 8)
            optimizer.zero_grad()
            outputs = model(data_batch)
            loss = criterion(outputs, target_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        print(f"{epoch + 1}/{epochs}: train loss - {running_loss / len(train_loader):.4f}, "
              f"validation loss - {val_loss:.4f}, validation accuracy - {val_accuracy:.4f}")


def validate_model(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data_batch, target_batch in loader:
            data_batch, target_batch = data_batch.to(device), target_batch.to(device)
            data_batch = data_batch.view(-1, 1, 8, 8)
            outputs = model(data_batch)
            loss = criterion(outputs, target_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += target_batch.size(0)
            correct += (predicted == target_batch).sum().item()
    return val_loss / len(loader), correct / total


def test_model(model, test_loader, device):
    model.eval()
    labels_predicted = []
    labels_true = []
    with torch.no_grad():
        for data_batch, target_batch in test_loader:
            data_batch = data_batch.to(device).view(-1, 1, 8, 8)
            outputs = model(data_batch)
            _, predicted = torch.max(outputs, 1)
            labels_predicted.extend(predicted.cpu().numpy())
            labels_true.extend(target_batch.numpy())

    conf_matrix = confusion_matrix(labels_true, labels_predicted)
    print(conf_matrix)
    accuracy = accuracy_score(labels_true, labels_predicted)
    print(f"Test accuracy: {accuracy:.4f}")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def vgg():
    digits = load_digits()
    data = digits.data
    target = digits.target

    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    train_size = 0.6
    val_size = 0.1

    data_size = len(data)
    train_end = int(train_size * data_size)
    val_end = train_end + int(val_size * data_size)

    data_train = data[:train_end]
    target_train = target[:train_end]

    data_val = data[train_end:val_end]
    target_val = target[train_end:val_end]

    data_test = data[val_end:]
    target_test = target[val_end:]

    data_train = torch.tensor(data_train, dtype=torch.float32)
    data_val = torch.tensor(data_val, dtype=torch.float32)
    data_test = torch.tensor(data_test, dtype=torch.float32)
    target_train = torch.tensor(target_train, dtype=torch.long)
    target_val = torch.tensor(target_val, dtype=torch.long)
    target_test = torch.tensor(target_test, dtype=torch.long)

    batch_size = 24
    train_loader = DataLoader(TensorDataset(data_train, target_train), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(data_val, target_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(data_test, target_test), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=30)
    test_model(model, test_loader, device)


if __name__ == '__main__':
    set_seed(15)
    vgg()
