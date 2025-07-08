import torch
import numpy as np
from skimage.draw import ellipse
from tqdm import tqdm

def ellipses_overlap(c1, axes1, c2, axes2, margin=0.0):
    """
    Check if two ellipses overlap approximately.
    c1, c2: centers (y, x)
    axes1, axes2: (axis_y, axis_x)
    margin: extra spacing margin
    """
    dist = np.linalg.norm(np.array(c1) - np.array(c2))
    radius1 = max(axes1)
    radius2 = max(axes2)
    return dist < (radius1 + radius2 + margin)

def generate_random_phantom(size=64, num_ellipses=6, max_attempts=100):
    image = np.zeros((size, size), dtype=np.float32)
    ellipses = []  # store tuples of (center, axes)

    for _ in range(num_ellipses):
        for attempt in range(max_attempts):
            center_y = np.random.randint(int(size * 0.25), int(size * 0.75))
            center_x = np.random.randint(int(size * 0.25), int(size * 0.75))
            axis_y = np.random.randint(int(size * 0.05), int(size * 0.3))
            axis_x = np.random.randint(int(size * 0.05), int(size * 0.3))

            # Check overlap with all previously placed ellipses
            overlap = False
            for (c, a) in ellipses:
                if ellipses_overlap((center_y, center_x), (axis_y, axis_x), c, a, margin=2):
                    overlap = True
                    break

            if not overlap:
                # Accept this ellipse
                ellipses.append(((center_y, center_x), (axis_y, axis_x)))
                break
        else:
            # Could not find non-overlapping ellipse in max_attempts, skip
            continue

        angle = np.random.uniform(0, 180)
        intensity = np.random.uniform(0.0, 1.0)

        rr, cc = ellipse(center_y, center_x, axis_y, axis_x, rotation=np.deg2rad(angle), shape=image.shape)
        image[rr, cc] += intensity

    #Ensure that the image is normalized to 0-1
    image = np.clip((image - image.min()) / (image.max() - image.min() + 1e-8), 0, 1)
    return image

def generate_phantom_dataset(num_samples=10000, size=64, num_ellipses=6):
    phantom_list = []
    for _ in tqdm(range(num_samples), desc="Generating random phantoms"):
        phantom = generate_random_phantom(size, num_ellipses)
        phantom_list.append(phantom)

    phantom_tensor = torch.tensor(np.stack(phantom_list), dtype=torch.float32)
    return phantom_tensor

# Example usage
phantoms = generate_phantom_dataset(num_samples=10000, size=128)
print(f"Generated phantoms tensor shape: {phantoms.shape}")
torch.save(phantoms, "data/phantoms.pt")
