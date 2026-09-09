total_train = total_train_df.shape[0]
total_validation = total_validation_df.shape[0]
total_test = test_df.shape[0]

print("We have %d training data, %d validation data and %d test data, both labels combined." %
      (total_train, total_validation, total_test))
