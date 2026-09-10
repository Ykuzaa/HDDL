def encode_localization(boxes):
    """Turn the list of objects of an image into a label of size 9."""
    label = np.zeros(9, dtype="f")

    if len(boxes) == 0:
        # No object on this image: presence stays at 0, and so does the rest
        return label

    # Object whose bounding box takes up the largest area
    class_index, cx, cy, width, height = max(boxes, key=lambda box: box[3] * box[4])

    label[0] = 1
    label[1:5] = (cx, cy, width, height)
    label[5:] = one_hot(class_index)

    return label