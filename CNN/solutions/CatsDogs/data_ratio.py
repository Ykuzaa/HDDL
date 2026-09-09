ratio_train = np.sum(train_categories)/total_train
ratio_validation = np.sum(validation_categories)/total_validation
ratio_test = np.sum(test_categories)/total_test

print("Train ratio dog/cat:", ratio_train)
print("Validation ratio dog/cat:", ratio_validation)
print("Test ratio dog/cat:", ratio_test)
