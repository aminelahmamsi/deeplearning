import time
import torch
from library.get_id import get_unique_run_id


def train_model(model, train_loader, val_loader, criterion, optimizer, device, hyperParameters, no_print=True):
    # Reset metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    timings = []

    # Unique run ID
    run_id = get_unique_run_id()

    for epoch in range(hyperParameters.epochs):
        start_time = time.time()
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Epoch timing
        epoch_time = time.time() - start_time
        timings.append(epoch_time)

        if not no_print:
            print(
                f"Epoch [{epoch + 1}/{hyperParameters.epochs}], "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                f"Time: {epoch_time:.2f}s"
            )

    return {
        "run_id": run_id,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "timings": timings,
    }
