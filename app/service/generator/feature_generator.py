from app.config.logging_config import logger
from app.optimization.sa_bbo import sa_bbo


def datagen():
    features = []
    labels = []
    input_shape = (256, 256, 3)

    # Dilation and Learning rates for parameter tuning in FCN
    lb = [2, 0.0001]
    ub = [7, 0.001]
    pop_size = 3
    prob_size = len(lb)
    epochs = 100

    best_solution, best_fitness = sa_bbo(lb, ub, pop_size, prob_size, epochs)
    dilation_rate = best_solution[0].astype('int32')
    learning_rate = best_solution[1]

    # FCN Model


class DataLoader:
    def __init__(self, num_classes=2):
        super().__init__()