import torch

def evaluate(model, test_loader, device, no_print=True):
    model.eval()
    
    TP = TN = FP = FN = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Accuracy counters
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Confusion matrix elements
            TP += ((predicted == 1) & (labels == 1)).sum().item()
            TN += ((predicted == 0) & (labels == 0)).sum().item()
            FP += ((predicted == 1) & (labels == 0)).sum().item()
            FN += ((predicted == 0) & (labels == 1)).sum().item()

    test_acc = 100 * correct / total

    if not no_print:
        print(
            f"TP:{TP}, "
            f"TN:{TN}, "
            f"FP:{FP}, "
            f"FN:{FN}, "
            f"TP:{TP}, "
            f"Test Accuracy: {test_acc:.2f}%"
        )

    return {
            "TP": TN,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "test_acc": test_acc,
        }