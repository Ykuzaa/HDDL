colorization_model = ColorizationModel(latent_dim=128)
colorization_encoder = train_ssl_model(colorization_model, 
                                       svhn_train_loader, 
                                       svhn_test_loader, 
                                       criterion=nn.MSELoss(), 
                                       optimizer=optim.Adam(colorization_model.parameters(), lr=0.001)
                                       )