(x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(x, y, groups)

# Statistics of the boxes, computed on the training images containing an object
present = y_train[:, 0] == 1
y_mean = y_train[present].mean(axis=0)
y_std = y_train[present].std(axis=0)

# Center-reduce the coordinates of the three sets
for y_set in (y_train, y_val, y_test):
    y_set[:, 1:5] = (y_set[:, 1:5] - y_mean[1:5]) / y_std[1:5]

x_train.shape, x_val.shape, x_test.shape