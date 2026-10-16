import inspect
from . import models as models

classes = [cls for name, cls in inspect.getmembers(models, inspect.isclass) if cls.__module__ == models.__name__]

model_list = {
    model.__name__: model for model in classes
}