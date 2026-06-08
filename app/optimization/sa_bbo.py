# Self-Adaptive Brown-Bear Optimization Algorithm
import glob
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.model.fcn import fcn_model, apply_mask
from skimage.metrics import peak_signal_noise_ratio
import cv2

logger = logging.getLogger("feature_pipeline")

def process_image(img_path, model):
    try:
        original_image = cv2.imread(img_path)

        if original_image is None:
            return None

        original_image = cv2.resize(original_image, (256, 256))

        prediction = model.predict(np.expand_dims(original_image, axis=0))
        segmented = apply_mask(original_image, prediction[0])
        segmented = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)

        return peak_signal_noise_ratio(original_image, segmented)
    except Exception as e:
        print(f" Error processing {img_path}: {e}")
        return None

def objective_func_1(x, input_shape, reference_images):
    val = max(1, x[0].astype('int32'))
    dilation_rate = (val, val)
    learning_rate = x[1]

    model = fcn_model(input_shape, dilation_rate, learning_rate)

    if not reference_images:
        print("No validation images found.")
        return float('inf')  # Return poor score

    psnr_scores = []

    # for reference_image in reference_images:
    #     psnr_scores.append(process_image(reference_image, model))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_image, reference_image, model) for reference_image in reference_images]
        for future in futures:
            result = future.result()
            if result is not None:
                psnr_scores.append(result)

    # If no valid PSNRs, return bad score
    if not psnr_scores:
        return float('inf')

    return 1 / np.mean(psnr_scores)

def objective_func(x, input_shape, reference_images):
    # dilation_rate = x[0].astype('int32')
    # learning_rate = x[1]
    # input_shape = (256, 256, 3)

    val = max(1, x[0].astype('int32'))
    dilation_rate = (val, val)
    learning_rate = x[1]

    original_image = cv2.imread(str('/data/raw/images/chest/cts/train/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/000017.png'))
    original_image = cv2.resize(original_image, (256, 256))

    model = fcn_model(input_shape, dilation_rate, learning_rate)
    # Predict segmentation mask using the FCN model
    segmentation_mask = model.predict(np.expand_dims(original_image, axis=0))

    # Apply segmentation mask to input image
    segmented_image = apply_mask(original_image, segmentation_mask[0])
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)

    psnr = peak_signal_noise_ratio(original_image, segmented_image)

    fit = 1 / psnr

    return fit

def sa_bbo(lb, ub, pop_size, prob_size, epochs, input_shape, reference_images):
    population = np.round(np.random.uniform(lb, ub, size=(pop_size, prob_size)), 4)
    logger.info(f"Pushpinder Features in SA-BBO: Ref Images Length- {len(reference_images)}")

    best_solution = population[0].copy()
    best_fitness = objective_func(best_solution, input_shape, reference_images)

    lb = np.array(lb)
    ub = np.array(ub)

    w_min = 0.1
    w_max = 1
    pi = np.pi

    logger.info(f"Pushpinder Features in SA-BBO: best_solution- {best_solution} for best_fitness- {best_fitness}")
    count = 0;
    for epoch in range(epochs):
        theta_k = epoch / epochs
        population = np.clip(population, lb, ub)
        for j in range(pop_size):

            fitness = objective_func(population[j], input_shape, reference_images)
            count +=1

            logger.info(f"Pushpinder Features: best_solution- {best_solution} for best_fitness- {best_fitness}, counter: {count}")

            if fitness < best_fitness:
                best_solution = population[j].copy()
                best_fitness = fitness

            # pedal scent marking behaviour
            if 0 < theta_k <= epochs / 3:
                # Update based on characteristic gait while walking
                alpha_k = np.random.uniform(0, 1)
                population[j] = population[j] - (theta_k * alpha_k * population[j])

            elif epochs / 3 < theta_k <= 2 * epochs / 3:
                # Update based on careful stepping characteristic
                beta_k = np.random.uniform(0, 1)
                f_k = beta_k * theta_k
                beta2_k = np.random.uniform(0, 1)
                l_k = np.round(1+beta2_k)

                # Improvement ------> Inertia Weight
                w = w_max - (w_max - w_min)*(epoch / epochs)
                population[j] = population[j] + w + f_k * (best_solution - l_k * population[j])

            else:
                # Update based on twisting feet characteristic
                gamma = np.random.uniform(0, 1)

                # Improvement ------> velocity controlling parameter
                x = (2 / (abs(2 - theta_k - np.sqrt(theta_k ** 2) - 4 * theta_k)))
                angular_velocity = 2 * pi * theta_k * gamma * x

                if best_solution is not None:
                    population[j] = (
                            population[j]
                            + angular_velocity * (best_solution - abs(population[j]))
                            - angular_velocity * (population[j] - abs(population[j]))
                    )
                # population[j] = population[j] + angular_velocity * (best_solution - abs(population[j])) - angular_velocity * (population[j] - abs(population[j]))

            # sniffing behaviour
            if np.random.rand() < objective_func(population[j], input_shape, reference_images):
                lamb = np.random.uniform(0, 1)
                population[j] = population[j] + lamb * (np.max(population) - np.min(population))

    return best_solution, best_fitness