EPOCHS = 2
LEARNING_RATE = 1e-3

optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss().to(device)

for epoch in range(EPOCHS):

    train_loss, train_acc = train_epoch(classifier, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(classifier, val_loader, criterion)

    epoch_time = time.time()

    print("")
    print(f'Epoch: {epoch+1:02} | Time: {epoch_time}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')