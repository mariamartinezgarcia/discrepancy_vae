import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pickle


# Define parameters
IMAGE_SIZE = (24, 24)  # 24x24 pixels to better center shapes
OUTPUT_FOLDER = "toy_images"

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

latdim = 4 # Number of nodes
samples = 2048
dim = 24*24*3 # Image size

# Define quarters and probability distributions
quarters = ["top-left", "top-right", "bottom-left", "bottom-right"]
quarter_probabilities = [0.25, 0.25, 0.25, 0.25]  # Uniform distribution

# Shape probability distribution per quarter (P(Shape | Position))
# We define the distribution over all available shapes: "cross", "square", "point", "stair", "L"
shape_distributions = {
    "top-left":{"cross": 0.5, "square": 0.5, "point": 0.0, "stair": 0.0, "L": 0.0},
    "top-right":{"cross": 0.0, "square": 0.0, "point": 0.7, "stair": 0.3, "L": 0.0},
    "bottom-left":{"cross": 0.3, "square": 0.0, "point": 0.0, "stair": 0.0, "L": 0.7},
    "bottom-right": {"cross": 0.0, "square": 0.2, "point": 0.0, "stair": 0.8, "L": 0.0},
}

# Color probability distribution per quarter (P(Color | Position))
# We define the distribution over all available shapes: "red", "green", "blue", "yellow", "magenta", "cyan", "white"
color_distributions = {
    "top-left": {"red":0.7, "green":0.3, "blue":0.0, "yellow":0.0, "magenta":0.0, "cyan":0.0, "white":0.0},
    "top-right": {"red":0.0, "green":0.0, "blue":0.5, "yellow":0.5, "magenta":0.0, "cyan":0.0, "white":0.0},
    "bottom-left": {"red":0.0, "green":0.0, "blue":0.0, "yellow":0.0, "magenta":0.6, "cyan":0.4, "white":0.0},
    "bottom-right": {"red":0.0, "green":0.0, "blue":0.0, "yellow":0.0, "magenta":0.0, "cyan":0.0, "white":1.0},
}
colors = {
    "red":      (1, 0, 0),
    "green":    (0, 1, 0), 
    "blue":     (0, 0, 1), 
    "yellow":   (1, 1, 0), 
    "magenta":  (1, 0, 1), 
    "cyan":     (0, 1, 1), 
    "white":    (1, 1, 1),
}

# Size probability distribution per color P(Size | Color)
# We define the distribution over all available sizes: "small", "large"
size_distributions = {
    "red": {"small": 0.9, "large": 0.1},
    "green": {"small": 0.1, "large": 0.9},
    "blue": {"small": 0.9, "large": 0.1},
    "yellow": {"small": 0.1, "large": 0.9},
    "magenta": {"small": 0.9, "large": 0.1},
    "cyan": {"small": 0.1, "large": 0.9},
    "white": {"small": 0.9, "large": 0.1},
}

# Helper function to sample from a probability distribution
def sample_from_distribution(distribution):
    items, probabilities = zip(*distribution.items())
    return random.choices(items, probabilities)[0]

# Function to compute shape center coordinates
def get_center_coords(quarter):
    if quarter == "top-left":
        return 2, 2  # Center at (5,5) inside the quarter
    elif quarter == "top-right":
        return 12, 2  # Center at (15,5)
    elif quarter == "bottom-left":
        return 2, 12  # Center at (5,15)
    elif quarter == "bottom-right":
        return 12, 12  # Center at (15,15)

# Function to compute shape placement coordinates
def get_shape_coords(quarter, size):
    """Returns the top-left coordinate for a shape in the given quarter."""
    half_quarter = 12 // 2  # 6 pixels for small, 12 pixels for large
    if quarter == "top-left":
        return (half_quarter - size // 2, half_quarter - size // 2)
    elif quarter == "top-right":
        return (12 + half_quarter - size // 2, half_quarter - size // 2)
    elif quarter == "bottom-left":
        return (half_quarter - size // 2, 12 + half_quarter - size // 2)
    elif quarter == "bottom-right":
        return (12 + half_quarter - size // 2, 12 + half_quarter - size // 2)
    
# Function to place the shape
def generate_mask(image, shape, quarter, color, shape_size):
    size = 6 if shape_size == "small" else 12  # Set shape size
    cx, cy = get_shape_coords(quarter, size)  # Get top-left coordinates
    mask = np.zeros(image.shape)

    if shape == "square":  # Hollow square
        mask[cy, cx:cx+size] = color  # Top edge
        mask[cy+size-1, cx:cx+size] = color  # Bottom edge
        mask[cy:cy+size, cx] = color  # Left edge
        mask[cy:cy+size, cx+size-1] = color  # Right edge

    if shape == "cross":  # Fully symmetrical cross
        mid_x = cx + size // 2
        mid_y = cy + size // 2
        arm_length = (size // 2) - 1  # Prevent out-of-bounds error

        for i in range(-arm_length, arm_length + 1):
            if 0 <= mid_y + i < 24:  # Vertical line (bounds check)
                mask[mid_y + i, mid_x] = color  
            if 0 <= mid_x + i < 24:  # Horizontal line (bounds check)
                mask[mid_y, mid_x + i] = color  
    elif shape == "point":
        mask[cy:cy+size, cx:cx+size] = color  
        #mask[cy + size // 2, cx + size // 2] = color  # Single point

    elif shape == "stair":  # Triangular staircase
        for i in range(size):
            for j in range(i + 1):
                mask[cy + i, cx + j] = color

    elif shape == "L":
        for i in range(size):
            mask[cy + i, cx] = color  # Vertical part
            mask[cy + size - 1, cx + i] = color  # Horizontal part
    
    return mask

# GENERATE DATA

metadata = {}

# single node
x = np.zeros((samples*latdim,dim))
xc = np.zeros((samples*latdim,dim))
c = ['' for _ in range(samples*latdim)]

# Generate control data (without interventions) and interventions
for i in range(latdim*samples):
    # No intervention
    image = np.ones((*IMAGE_SIZE, 3))  # Black background

    # Step 1: Sample the quarter (position)
    quarter = random.choices(quarters, quarter_probabilities)[0]

    # Step 2: Sample the shape based on the chosen quarter
    shape = sample_from_distribution(shape_distributions[quarter])

    # Step 3: Sample the color based on the chosen quarter
    color = sample_from_distribution(color_distributions[quarter])
    rgb = colors[color]

    # Step 4: Sample the color based on the chosen quarter
    size = sample_from_distribution(size_distributions[color])

    # Step 5: Place the shape in the correct quarter
    mask = generate_mask(image, shape, quarter, rgb, size)
    image = image*mask
    # Show image
    #plt.imshow(image)
    #plt.show()
    # Save image
    img_filename = f"{OUTPUT_FOLDER}/img_ctrl{i}.png"
    plt.imsave(img_filename, image)

    x[i, :] = image.flatten()

    # Store metadata
    metadata[f"img_ctrl_{i}"] = {"position": quarter, "shape": shape, "color": color, "size": size}

idx = 0
for j in range(latdim):
    # Intervene node j
    for i in range(samples):

        image = np.ones((*IMAGE_SIZE, 3))  # Black background
    
        # Step 1: Sample the quarter (position)
        if j==0:
            c[idx+i] = 'position'
        quarter = random.choices(quarters, quarter_probabilities)[0]

        if j==1:
            shapes = ["cross", "square", "point", "stair", "L"]
            shapes_probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
            shape = random.choices(shapes, shapes_probabilities)[0]
            c[idx+i] = 'shape'
        else:
            # Step 2: Sample the shape based on the chosen quarter
            shape = sample_from_distribution(shape_distributions[quarter])

        if j==2:
            color_list = ["red", "green", "blue", "yellow", "magenta", "cyan", "white"]
            colors_probabilities = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
            color = random.choices(color_list, colors_probabilities)[0]
            rgb = colors[color]
            c[idx+i] = 'color'
        else:
            # Step 3: Sample the color based on the chosen quarter
            color = sample_from_distribution(color_distributions[quarter])
            rgb = colors[color]

        if j==3:
            sizes = ["small", "large"]
            sizes_probabilities = [0.5, 0.5]
            size = random.choices(sizes, sizes_probabilities)[0]
            c[idx+i] = 'size'
        else:
            # Step 4: Sample the color based on the chosen quarter
            size = sample_from_distribution(size_distributions[color])

        # Step 5: Place the shape in the correct quarter
        mask = generate_mask(image, shape, quarter, rgb, size)
        image = image*mask
        # Show image
        #plt.imshow(image)
        #plt.show()
        # Save image
        img_filename = f"{OUTPUT_FOLDER}/img_ptbs_c{j}_{i}.png"
        plt.imsave(img_filename, image)

        xc[i, :] = image.flatten()

        # Store metadata
        metadata[f"img_ptbs_c{j}_{i}"] = {"position": quarter, "shape": shape, "color": color, "size": size}

    idx += samples


dataset = {}
dataset['ptb_targets'] = ['position', 'shape', 'color', 'size']
dataset['single'] = {'X':x, 'Xc':xc, 'ptbs':c}


with open(f'./data_toy_images.pkl', 'wb') as f:
	pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
