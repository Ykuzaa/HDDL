model = LocalizationNetVGG(make_conv_base()).to(device)
batch_size = 18
epochs = 30

loss_weights = {'p': 1, 'coord': 5, 'classes': 1}

train_loader, val_loader = make_loaders(x_train, y_train, x_val, y_val, batch_size)

# --- #

print("Transfer learning")
for parameter in model.conv_base.parameters():
    parameter.requires_grad = False

optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
history = fit_localization(model, train_loader, val_loader, optimizer,
                           losses, loss_weights, epochs)

# --- #

print("\nFine tuning")
for parameter in model.conv_base.parameters():
    parameter.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
history = fit_localization(model, train_loader, val_loader, optimizer,
                           losses, loss_weights, epochs)