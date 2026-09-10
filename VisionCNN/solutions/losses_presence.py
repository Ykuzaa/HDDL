def coord_loss_masked(pred, target, presence):
    """Mean square error, restricted to the images that contain an object."""
    # One value per image, of shape (N, 1)
    per_image = ((pred - target) ** 2).mean(dim=1, keepdim=True)
    return (presence * per_image).sum() / presence.sum().clamp(min=1)


def class_loss_masked(pred, target, presence):
    """Cross-entropy, restricted to the images that contain an object."""
    # One value per image, of shape (N, 1)
    per_image = F.cross_entropy(pred, target, reduction='none').unsqueeze(1)
    return (presence * per_image).sum() / presence.sum().clamp(min=1)


# The presence loss is the one of Part I: it concerns every image
losses_presence = {'p': presence_loss, 'coord': coord_loss_masked, 'classes': class_loss_masked}