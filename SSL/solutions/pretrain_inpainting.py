inpainting_model = InpaintingModel(mask_size=8)
inpainting_encoder = train_ssl_model(inpainting_model,
                                     svhn_train_loader, 
                                     svhn_test_loader, 
                                     criterion=nn.MSELoss(), 
                                     optimizer=optim.Adam(inpainting_model.parameters(), lr=0.001)
                                     )