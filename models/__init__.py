from models import cnn_model

def create_model(opts):
    if opts.model_type == 'model_cnn':
        model = cnn_model.CNNModel(opts)

    else:
        raise NotImplementedError

    return model
