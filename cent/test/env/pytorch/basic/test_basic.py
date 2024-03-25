import pytest 
import torch

from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"

@pytest.mark.env
def test_pytorch_install():
    ver = torch.__version__
    major = ver.split(".")[0]
    assert int(major) >= 1


@pytest.mark.env
def test_pytorch_v2():
    ver = torch.__version__
    major = ver.split(".")[0]
    assert int(major) >=  2

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

fmnist_path = "D:/datasets/fmnist"

@pytest.mark.app
def test_mnist():

    training_data = datasets.FashionMNIST(
        root=fmnist_path,
        train=True,
        download=False,
        transform=ToTensor(),
    )

    # Download test data from open datasets
    test_data = datasets.FashionMNIST(
        root=fmnist_path,
        train=False,
        download=False,
        transform=ToTensor(),
    )

    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # validate the data structure
    for X, y in test_dataloader:
        assert X.shape[0] == batch_size
        assert X.shape[1] == 1
        assert X.shape[2] == 28
        assert X.shape[3] == 28
        assert y.shape[0] == batch_size
        assert y.dtype == torch.int64
        break
    
    # creating model


    assert device == "cuda"

    ## define model

    model = NeuralNetwork().to(device)
    # print(model)

    # optimizing the model parameters

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    model_path = "D:/checkpoints/fmnist/pytorch_hello_5.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch Model State to {model_path}")

    assert True 

@pytest.mark.app
def test_load_pth():
    model = NeuralNetwork().to(device)
    model_path = "D:/checkpoints/fmnist/pytorch_hello_5.pth"
    model.load_state_dict(torch.load(model_path))

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    test_data = datasets.FashionMNIST(
        root=fmnist_path,
        train=False,
        download=False,
        transform=ToTensor(),
    )

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

    assert True