# Definition of the YOLO loss function
def YOLOss(lambda_coord, lambda_noobj, batch_size):

    # "Green" part: subpart concerning the confidence index
    # and the class probabilities in the case where a box is present in the cell
    def box_loss(y_pred, y_true):
        return (((y_true[:, 0] - torch.sigmoid(y_pred[:, 0])) ** 2).sum()
                + ((y_true[:, 5:9] - torch.softmax(y_pred[:, 5:9], dim=1)) ** 2).sum())

    # "Blue" part: subpart concerning the coordinates of the bounding box in the case where a box is present in the cell
    def coord_loss(y_pred, y_true):
        return ((y_true[:, 1:5] - y_pred[:, 1:5]) ** 2).sum()

    # "Red" part: subpart concerning the confidence index in case no box is present in the cell
    def nobox_loss(y_pred, y_true):
        return ((y_true[:, 0] - torch.sigmoid(y_pred[:, 0])) ** 2).sum()

    def YOLO_loss(y_pred, y_true):

        # Reshape the tensors from bs x S x S x (5B+C) to (bsxSxS) x (5B+C)
        y_true = y_true.reshape(-1, 9)
        y_pred = y_pred.reshape(-1, 9)

        # Search (in y_true labels) for the cells for which at least the first bounding box is present
        not_empty = y_true[:, 0] >= 1
        empty = ~not_empty

        # Separate the cells of y_true and y_pred with or without bounding box
        y_true_notempty, y_pred_notempty = y_true[not_empty], y_pred[not_empty]
        y_true_empty, y_pred_empty = y_true[empty], y_pred[empty]

        return (box_loss(y_pred_notempty, y_true_notempty)
                + lambda_coord * coord_loss(y_pred_notempty, y_true_notempty)
                + lambda_noobj * nobox_loss(y_pred_empty, y_true_empty)) / batch_size

    # Return a function
    return YOLO_loss