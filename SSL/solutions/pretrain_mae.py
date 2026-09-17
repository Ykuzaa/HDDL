mae_model = MaskedAutoencoderModel(latent_dim=128, mask_ratio=1/4)
mae_encoder = train_ssl_model(mae_model,
                              svhn_train_loader, 
                              svhn_test_loader, 
                              criterion=nn.MSELoss(), 
                              optimizer=optim.Adam(mae_model.parameters(), lr=0.001)
                              )