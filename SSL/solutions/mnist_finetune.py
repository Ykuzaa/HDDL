# Fine-tuning loop for classification
def fine_tune_classification(encoder, train_loader, test_loader, epochs=5, encoder_in_channels=3):
    model = Classifier(encoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Freeze the encoder's weights
    for param in model.encoder.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            if encoder_in_channels == 3:
                images = torch.cat((images, images, images), dim=1)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Compute accuracy
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}, Accuracy: {100 * correct / total:.2f}%")

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if encoder_in_channels == 3:
                images = torch.cat((images, images, images), dim=1)
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')