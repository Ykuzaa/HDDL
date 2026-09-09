opt = Adam(learning_rate=3e-4)
model = create_model_localization_VGG()
batch_size = 18
epochs = 30

loss = ['binary_crossentropy', 'mse', 'categorical_crossentropy']
metrics = ['accuracy', iou(), 'accuracy']
loss_weights = [1, 5, 1]

# --- #

print("Transfert Learning")
conv_base.trainable = False
opt = Adam(learning_rate=3e-4)
model.compile(loss = loss,
              optimizer = opt,
              metrics = metrics,
              loss_weights = loss_weights)

history = model.fit(x_train, [y_train[:,0], y_train[:,1:5], y_train[:,5:9]],
              epochs = epochs,
              batch_size = batch_size,
              validation_data = (x_val, [y_val[:,0], y_val[:,1:5], y_val[:,5:9]]))

# --- #
print("\n")
# --- #

print("Fine tuning")
conv_base.trainable = True
opt = Adam(learning_rate=1e-6)
model.compile(loss = loss,
              optimizer = opt,
              metrics = metrics,
              loss_weights = loss_weights)

history = model.fit(x_train, [y_train[:,0], y_train[:,1:5], y_train[:,5:9]],
              epochs = epochs,
              batch_size = batch_size,
              validation_data = (x_val, [y_val[:,0], y_val[:,1:5], y_val[:,5:9]]))