x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

# To improve the training, we can center-reduce the coordinates of the bounding boxes
y_std = np.std(y_train, axis=0)
y_mean = np.mean(y_train, axis=0)
y_train[:,1:5] = (y_train[:,1:5] - y_mean[1:5])/y_std[1:5]
y_val[:,1:5] = (y_val[:,1:5] - y_mean[1:5])/y_std[1:5]

# And normalize the color values
x_train = x_train/255
x_val = x_val/255