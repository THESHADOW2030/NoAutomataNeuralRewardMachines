gnn_registry = {}

def get_class1( kls ):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)

    return m


def get_class(path):
    module_path, class_name = path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)

def register(id="", entry_point=None, **kwargs):
    gnn_registry[id] = {
        "class": get_class(entry_point),
        "kwargs": kwargs
    }

def lookup(gnn_id):
    return gnn_registry[gnn_id]
