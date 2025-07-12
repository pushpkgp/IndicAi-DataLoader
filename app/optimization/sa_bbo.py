# Self Adaptive Brown-Bear Optimization Algorithm
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from app.model.fcn import fcn_model, apply_mask
from skimage.metrics import peak_signal_noise_ratio
import cv2
from app.config.config import Config

def process_image(img_path, model):
    try:
        original_image = cv2.imread(img_path)
        if original_image is None:
            return None
        original_image = cv2.resize(original_image, (256, 256))

        prediction = model.predict(np.expand_dims(original_image, axis=0))
        segmented = apply_mask(original_image, prediction[0])
        segmented = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)

        psnr = peak_signal_noise_ratio(original_image, segmented)
        return psnr
    except Exception as e:
        print(f" Error processing {img_path}: {e}")
        return None

def objective_func(x):
    dilation_rate = x[0].astype('int32')
    learning_rate = x[1]
    input_shape = (256, 256, 3)

    # Directory containing validation images
    val_dir = Config().get_path("val_images")


    # Get list of image file paths (first 10 only)
    image_files = sorted([os.path.join(val_dir, f) for f in os.listdir(val_dir)
                   if f.endswith('.png') or f.endswith('.jpg')])[:10]

    if not image_files:
        print("No validation images found.")
        return float('inf')  # Return poor score

    model = fcn_model(input_shape, dilation_rate, learning_rate)

    psnr_scores = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_image, path, model) for path in image_files]
        for future in futures:
            result = future.result()
            if result is not None:
                psnr_scores.append(result)

    # If no valid PSNRs, return bad score
    if not psnr_scores:
        return float('inf')

    return 1 / np.mean(psnr_scores)

def sa_bbo(lb, ub, pop_size, prob_size, epochs):
    population = np.random.uniform(lb, ub, size=(pop_size, prob_size))
    lb = np.array(lb)
    ub = np.array(ub)
    best_solution = None
    best_fitness = float('inf')
    w_min = 0.1
    w_max = 1
    pi = np.pi
    for epoch in range(epochs):
        theta_k = epoch / epochs
        for j in range(pop_size):
            population[j, population[j] < lb] = lb[population[j] < lb]
            population[j, population[j] > ub] = ub[population[j] > ub]
            fitness = objective_func(population[j])

            if fitness < best_fitness:
                best_solution = population[j]
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
                population[j] = population[j] + angular_velocity * (best_solution - abs(population[j])) - angular_velocity * (population[j] - abs(population[j]))

            # sniffing behaviour
            if np.random.rand() < objective_func(population[j]):
                lamb = np.random.uniform(0, 1)
                population[j] = population[j] + lamb * (np.max(population) - np.min(population))

    return best_solution, best_fitness