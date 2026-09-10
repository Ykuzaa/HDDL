def run_epoch_yolo(model, loader, criterion, optimizer=None):
    """One pass over `loader`. If `optimizer` is None, the pass is a plain evaluation."""
    is_train = optimizer is not None
    model.train(is_train)

    totals = {name: 0.0 for name in YOLO_METRIC_NAMES}
    n_seen, n_occupied = 0, 0

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        with torch.set_grad_enabled(is_train):
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        metrics, n_batch_occupied = yolo_metrics(y_pred, y_batch)
        n_batch = x_batch.shape[0]
        n_seen += n_batch
        n_occupied += n_batch_occupied

        totals['loss'] += loss.item() * n_batch
        totals['p_accuracy'] += metrics['p_accuracy'] * n_batch
        totals['coord_iou'] += metrics['coord_iou'] * n_batch_occupied
        totals['classes_accuracy'] += metrics['classes_accuracy'] * n_batch_occupied

    counts = {'loss': n_seen, 'p_accuracy': n_seen,
              'coord_iou': n_occupied, 'classes_accuracy': n_occupied}
    return {name: totals[name] / max(counts[name], 1) for name in YOLO_METRIC_NAMES}


def fit_yolo(model, train_loader, val_loader, optimizer, criterion, epochs,
             checkpoint_path="yolo_best.pt"):
    """Train `model`, keeping a copy of the parameters of the best epoch."""
    history = {name: [] for name in YOLO_METRIC_NAMES}
    history.update({'val_' + name: [] for name in YOLO_METRIC_NAMES})
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_metrics = run_epoch_yolo(model, train_loader, criterion, optimizer=optimizer)
        val_metrics = run_epoch_yolo(model, val_loader, criterion, optimizer=None)

        for name in YOLO_METRIC_NAMES:
            history[name].append(train_metrics[name])
            history['val_' + name].append(val_metrics[name])

        print(f"Epoch {epoch+1}/{epochs}"
              f" - loss: {train_metrics['loss']:.4f} - val_loss: {val_metrics['loss']:.4f}"
              f" - val_presence: {val_metrics['p_accuracy']:.3f}"
              f" - val_IoU: {val_metrics['coord_iou']:.3f}"
              f" - val_class_acc: {val_metrics['classes_accuracy']:.3f}")

        # Save the model each time the validation loss reaches a new minimum
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    val_loss improved, model saved to {checkpoint_path}")

    print(f"\\nBest validation loss: {best_val_loss:.4f}")
    return history