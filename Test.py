import torch
from sklearn.metrics import f1_score, classification_report
import numpy as np


def test_model(model, data_loader, device,loss_values,ListofResult,dataset,hidden_size, num_epochs, batch, learning_rate, momentum):
    model.eval()
    y_pred = []
    y_test = []
    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        y_pred.append(model(data))
        y_test.append(targets)
    y_pred = torch.stack(y_pred).squeeze()
    y_test = torch.stack(y_test).squeeze()
    y_pred = y_pred.argmax(dim=1, keepdim=True)

    f1_score_macro = f1_score(y_test, y_pred, average="macro", labels=np.unique(y_pred))
    report = classification_report(y_test, y_pred, zero_division=1)

    resultdics = {
        "f1_score_macro": f1_score_macro,
        "report": report,
        "loss": loss_values,
        "hidden_size":hidden_size,
        "num_epochs":num_epochs,
        "batch":batch,
        "learning_rate":learning_rate,
        "momentum":momentum,
        "y_pred": y_pred,
        "y_test": y_test
    }

    ListofResult.append(resultdics)
    return ListofResult