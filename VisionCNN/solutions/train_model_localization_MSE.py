epochs = 30
batch_size = 15

model = LocalizationNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

train_loader, val_loader = make_loaders(x_train, y_train, x_val, y_val, batch_size)

# Same three outputs, but every loss is now a mean square error
def presence_loss_mse(pred, target, presence):
    return F.mse_loss(torch.sigmoid(pred), target)


def coord_loss_mse(pred, target, presence):
    return F.mse_loss(pred, target)


def class_loss_mse(pred, target, presence):
    return F.mse_loss(torch.softmax(pred, dim=1), target)


losses_mse = {'p': presence_loss_mse, 'coord': coord_loss_mse, 'classes': class_loss_mse}

loss_weights = {'p': 1, 'coord': 1, 'classes': 1}

history = fit_localization(model, train_loader, val_loader, optimizer,
                           losses_mse, loss_weights, epochs)