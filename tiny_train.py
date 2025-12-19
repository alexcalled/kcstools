import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # MNIST: 28x28 grayscale digits, labels 0-9
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # converts [0..255] PIL image to float tensor [0..1]
        ]
    )

    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

    # Tiny model: flatten 28*28 -> 10 classes
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    model.train()
    losses = []

    steps = 200
    it = iter(train_loader)

    for step in range(1, steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)

        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        if step == 1 or step % 20 == 0:
            print(f"step {step:03d} | loss {loss.item():.4f}")

    # Basic sanity check: average of last 20 should be lower than first 20 (usually true)
    first = sum(losses[:20]) / 20
    last = sum(losses[-20:]) / 20
    print(f"avg loss first 20: {first:.4f}")
    print(f"avg loss last  20: {last:.4f}")

    if last >= first:
        print("WARNING: loss did not decrease on average. Something may be wrong.")


if __name__ == "__main__":
    main()
