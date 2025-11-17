from SymGroundMultiTask.grounder_models import CNN_grounder, GridworldClassifier, ObjectCNN


grounder_models = ["ObjectCNN", "CNN_grounder", "GridworldClassifier"]


def make_grounder(model_name, num_symbols, obs_size, freeze_grounder=False):

    assert model_name in grounder_models or model_name == None

    if model_name == "ObjectCNN":
        model = ObjectCNN(obs_size, num_symbols)

    elif model_name == "CNN_grounder":
        model = CNN_grounder(num_symbols)

    elif model_name == "GridworldClassifier":
        model = GridworldClassifier(num_symbols)

    elif model_name == None:
        return None

    if freeze_grounder:
        for param in model.parameters():
            param.requires_grad = False

    return model