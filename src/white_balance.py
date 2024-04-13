import numpy as np

def white_balance(image, lam=0.2):
    mean_r = np.mean(image[:,:,2])
    mean_g = np.mean(image[:,:,1])
    mean_b = np.mean(image[:,:,0])

    mean = (mean_r + mean_g + mean_b) / 3

    #As suggested by the paper
    mean_increased = 128 + mean*lam

    corrected_r = (mean_increased / mean_r) * image[:,:,2]
    corrected_g = (mean_increased / mean_g) * image[:,:,1]
    corrected_b = (mean_increased / mean_b) * image[:,:,0]

    corrected_image = np.stack([corrected_b, corrected_g, corrected_r], axis=2)
    corrected_image = np.clip(corrected_image, 0, 255)
    corrected_image = corrected_image.astype(np.uint8)

    return corrected_image