

def train_model(model, criterion, optimizer, epochs, data_loader, device,scheduler,batch):
    model.train()
    loss_values = []

    """Early stopping"""
    the_last_loss = 100
    patience = 4
    if (batch <= 16):
        patience = 6
    trigger_times = 0

    for epoch in range(epochs):
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            # Forward pass
            y_pred = model(data)
            # Compute Loss
            loss = criterion(y_pred.squeeze(), targets)
            loss_values.append(loss.item())
            # Backward pass
            loss.backward()
            optimizer.step()
        scheduler.step()
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        print('Epoch {} train loss: {}'.format(epoch, loss.item()))

        """Early stopping"""
        the_current_loss = loss.item()

        if the_current_loss > the_last_loss:
            trigger_times += 1

            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                return model, loss_values

        else:
            trigger_times = 0

        the_last_loss = the_current_loss

    return model, loss_values