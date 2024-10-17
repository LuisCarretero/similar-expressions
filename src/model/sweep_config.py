sweep_configuration = {
    'method': 'grid',
    'metric': {
        'name': 'train/loss',
        'goal': 'minimize'
    },
    'parameters': {
        'model.decoder.rnn_type': {
            'values': ['lstm', 'mlp'],
        },
        'model.decoder.hidden_size': {
            'values': [64, 128, 256, 512]
        }
    }
}