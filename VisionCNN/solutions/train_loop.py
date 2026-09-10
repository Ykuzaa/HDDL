METRIC_NAMES = ['loss', 'p_loss', 'coord_loss', 'classes_loss',
                'p_accuracy', 'coord_iou', 'classes_accuracy']

# Metrics that only make sense on the images containing an object
ON_PRESENT_ONLY = {'coord_loss', 'classes_loss', 'coord_iou', 'classes_accuracy'}


def run_epoch(model, loader, losses, loss_weights, optimizer=None):
    """One pass over `loader`. If `optimizer` is None, the pass is a plain evaluation.

    Returns a dictionary giving the average of each metric over the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)

    totals = {name: 0.0 for name in METRIC_NAMES}
    n_seen, n_present = 0, 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        # The three targets, in the same order as the three outputs of the network
        t_p, t_coord, t_classes = y_batch[:, 0:1], y_batch[:, 1:5], y_batch[:, 5:9]

        with torch.set_grad_enabled(is_train):
            # Forward pass
            out_p, out_coord, out_classes = model(x_batch)

            # One loss per output. The presence is passed to all three: the last two
            # need it to leave out the images that contain no object
            l_p = losses['p'](out_p, t_p, t_p)
            l_coord = losses['coord'](out_coord, t_coord, t_p)
            l_classes = losses['classes'](out_classes, t_classes, t_p)

            # ... combined into the total loss
            loss = (loss_weights['p'] * l_p
                    + loss_weights['coord'] * l_coord
                    + loss_weights['classes'] * l_classes)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Accumulation of the metrics, weighted by the number of images they were computed on
        with torch.no_grad():
            n_batch = x_batch.shape[0]
            n_seen += n_batch
            totals['loss'] += loss.item() * n_batch
            totals['p_loss'] += l_p.item() * n_batch
            # An object is predicted as present when its score is positive, i.e. sigmoid(score) > 0.5
            totals['p_accuracy'] += ((out_p > 0).float() == t_p).float().mean().item() * n_batch

            # The box and the class only mean something on the images that contain an object
            present = t_p[:, 0] > 0.5
            n_batch_present = int(present.sum())
            if n_batch_present > 0:
                n_present += n_batch_present
                totals['coord_loss'] += l_coord.item() * n_batch_present
                totals['classes_loss'] += l_classes.item() * n_batch_present
                totals['coord_iou'] += iou_metric(out_coord[present], t_coord[present]) * n_batch_present
                totals['classes_accuracy'] += (out_classes[present].argmax(1)
                                               == t_classes[present].argmax(1)).float().mean().item() * n_batch_present

    # Each metric is divided by the number of images it was accumulated over
    counts = {name: (n_present if name in ON_PRESENT_ONLY else n_seen) for name in METRIC_NAMES}
    return {name: totals[name] / max(counts[name], 1) for name in METRIC_NAMES}


def fit_localization(model, train_loader, val_loader, optimizer, losses, loss_weights, epochs):
    """Train `model` for `epochs` epochs and return the history of the metrics."""
    history = {name: [] for name in METRIC_NAMES}
    history.update({'val_' + name: [] for name in METRIC_NAMES})

    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, losses, loss_weights, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, losses, loss_weights, optimizer=None)

        for name in METRIC_NAMES:
            history[name].append(train_metrics[name])
            history['val_' + name].append(val_metrics[name])

        print(f"Epoch {epoch+1}/{epochs}"
              f" - loss: {train_metrics['loss']:.4f} - val_loss: {val_metrics['loss']:.4f}"
              f" - val_IoU: {val_metrics['coord_iou']:.3f}"
              f" - val_class_acc: {val_metrics['classes_accuracy']:.3f}")

    return history