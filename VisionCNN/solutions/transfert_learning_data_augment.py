opt = Adam(learning_rate=3e-4)
model = create_model_localization()
batch_size = 18
epochs = 30

loss = ['binary_crossentropy', 'mse', 'categorical_crossentropy']
metrics = ['accuracy', iou(), 'accuracy']
loss_weights = [1, 5, 1]

train_gen = WildLifeSequence(x_train, y_train, batch_size, augmentations=AUGMENTATIONS_TRAIN)
valid_gen = WildLifeSequence(x_val, y_val, batch_size, augmentations=AUGMENTATIONS_TEST)

# --- #

print("Transfert Learning")
conv_base.trainable = False
opt = Adam(learning_rate=3e-4)
model.compile(loss = loss,
              optimizer = opt,
              metrics = metrics,
              loss_weights = loss_weights)

history = model.fit(train_gen,
              epochs = epochs,
              batch_size = batch_size,
              validation_data = valid_gen)

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

history = model.fit(train_gen,
              epochs = epochs,
              batch_size = batch_size,
              validation_data = valid_gen)