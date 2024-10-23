sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'name': 'train/loss',
        'goal': 'minimize'
    },
    'parameters': {
        'model.encoder.depth': {
            'values': [2, 3, 4]
        },
        'model.encoder.width': {
            'values': [128, 256, 512, 1024, 2048]
        },
        'model.value_decoder.depth': {
            'values': [2, 3, 4]
        },
        'model.value_decoder.width': {
            'values': [128, 256, 512, 1024, 2048]
        },
        'model.z_size': {
            'values': [128, 256, 512]
        },
    }
}