def create_model_localization_VGG(input_shape=(64, 64, 3)):

    input_layer = Input(shape=input_shape)
    vgg = conv_base(input_layer)
    x = Flatten()(vgg)

    output_p = Dense(1, activation='sigmoid', name='p')(x)
    output_coord = Dense(4, activation='linear', name='coord')(x)
    output_class = Dense(4, activation='softmax', name='classes')(x)

    output= [output_p, output_coord, output_class]
    model = Model(input_layer, output)

    return model