# Definition of the YOLO loss function
def YOLOss(lambda_coord, lambda_noobj, batch_size):

    # "Green" part: subpart concerning the confidence index
    # and the class probabilities probabilities in the case where a box is present in the cell
    def box_loss(y_true, y_pred):
        return tf.reduce_sum(tf.square(y_true[:,0] - tf.sigmoid(y_pred[:,0]))) + tf.reduce_sum(tf.square(y_true[:,5:9] - tf.nn.softmax(y_pred[:,5:9])))

    # "Blue" part: subpart concerning the coordinates of the bounding box in the case where a box is present in the cell
    def coord_loss(y_true, y_pred):
        return tf.reduce_sum(tf.square(y_true[:,1:5] - y_pred[:,1:5]))


    # "Red" part: subpart concerning the confidence index in case no box is present in the cell
    def nobox_loss(y_true, y_pred):
        return tf.reduce_sum(tf.square(y_true[:,0] - tf.sigmoid(y_pred[:,0])))


    def YOLO_loss(y_true, y_pred):

        # Reshape the tensors from bs x S x S x (5B+C) to (bsxSxS) x (5B+C)
        y_true = tf.reshape(y_true, shape=(-1, 9))
        y_pred = tf.reshape(y_pred, shape=(-1, 9))

        # Search (in y_true labels) for indices of cells for which at least the first bounding box is present
        not_empty = tf.greater_equal(y_true[:, 0], 1)
        indices = tf.range(0, tf.shape(y_true)[0], delta=1)
        indices_notempty_cells = indices[not_empty]

        empty = tf.less_equal(y_true[:, 0], 0)
        indices_empty_cells = indices[empty]

        # Separate the cells of y_true and y_pred with or without bounding box
        y_true_notempty = tf.gather(y_true, indices_notempty_cells)
        y_pred_notempty = tf.gather(y_pred, indices_notempty_cells)

        y_true_empty = tf.gather(y_true, indices_empty_cells)
        y_pred_empty = tf.gather(y_pred, indices_empty_cells)

        return (box_loss(y_true_notempty, y_pred_notempty) + lambda_coord*coord_loss(y_true_notempty, y_pred_notempty) + lambda_noobj*nobox_loss(y_true_empty, y_pred_empty))/batch_size


    # Return a function
    return YOLO_loss