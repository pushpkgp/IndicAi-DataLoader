import logging

from app.model.fcn import fcn_model
from app.optimization.sa_bbo import sa_bbo

logger = logging.getLogger("feature_pipeline")

def get_segmentation_mask_model(reference_images):
    input_shape = (256, 256, 3)
    logger.info(f"Pushpinder Segmentation")

    # Dilation and Learning rates for parameter tuning in FCN
    lb = [2, 0.0001]
    ub = [7, 0.001]
    pop_size = 3
    prob_size = len(lb)
    epochs = 100

    best_solution, best_fitness = sa_bbo(lb, ub, pop_size, prob_size, epochs, input_shape, reference_images)
    logger.info(f"---------->>> best_solution- {best_solution} for best_fitness- {best_fitness}")

    val = max(1, best_solution[0].astype('int32'))
    dilation_rate = (val, val)
    learning_rate = best_solution[1]
    return fcn_model(input_shape, dilation_rate, learning_rate)