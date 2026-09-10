def presence_loss(pred, target, presence):
    """Loss of the presence output. `pred` holds raw scores, `target` is 0 or 1."""
    return F.binary_cross_entropy_with_logits(pred, target)


def coord_loss(pred, target, presence):
    """Loss of the coordinates, both already center-reduced."""
    return F.mse_loss(pred, target)


def class_loss(pred, target, presence):
    """Loss of the classification. `pred` holds raw scores, `target` one-hot vectors."""
    return F.cross_entropy(pred, target)


losses = {'p': presence_loss, 'coord': coord_loss, 'classes': class_loss}