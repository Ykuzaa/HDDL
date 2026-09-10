def encode_yolo(boxes):
    """Turn the list of objects of an image into a (S, S, 5B + C) grid."""
    label = np.zeros((CELL_PER_DIM, CELL_PER_DIM, NB_CLASSES + 5 * BOX_PER_CELL), dtype="f")

    for class_index, cx, cy, width, height in boxes:
        # Indices of the cell the centre of the box falls into.
        # The min guards against a centre sitting exactly on the last edge.
        ind_x = min(int(cx * CELL_PER_DIM), CELL_PER_DIM - 1)
        ind_y = min(int(cy * CELL_PER_DIM), CELL_PER_DIM - 1)

        # YOLO: "The (x, y) coordinates represent the center of the box relative to the
        # bounds of the grid cell" -> coordinates of the centre inside its own cell
        cx_cell = cx * CELL_PER_DIM - ind_x
        cy_cell = cy * CELL_PER_DIM - ind_y

        # First free box of that cell; the object is dropped if the cell is already full
        ind_box = 0
        while ind_box < BOX_PER_CELL and label[ind_x, ind_y, 5 * ind_box] == 1:
            ind_box = ind_box + 1
        if ind_box == BOX_PER_CELL:
            continue

        label[ind_x, ind_y, 5 * ind_box] = 1
        label[ind_x, ind_y, 5 * ind_box + 1] = cx_cell
        label[ind_x, ind_y, 5 * ind_box + 2] = cy_cell
        label[ind_x, ind_y, 5 * ind_box + 3] = math.sqrt(width)
        label[ind_x, ind_y, 5 * ind_box + 4] = math.sqrt(height)

        # Class probabilities, shared by the cell and placed after the boxes
        label[ind_x, ind_y, 5 * BOX_PER_CELL:] = one_hot(class_index)

    return label