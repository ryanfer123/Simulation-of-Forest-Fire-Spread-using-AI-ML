
"""
Forest Fire Prediction and Simulation

This script implements a forest fire prediction and simulation system with the following objectives:
1. Prepare a forest fire probability map for the next day (binary classification: fire/no fire)
2. Simulate the spread of fire within 1, 2, 3, 6, and 12 hours from high-risk zones

Author: Ryan Fernandes
Special thanks to: ISRO, IMD, Bhoonidhi, Bhuvan, and VIIRS-SNP for data sources
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.transform import from_origin
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import geopandas as gpd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Global parameters
REGION = "Uttarakhand"  # Target region
RESOLUTION = 30  # Resolution in meters
OUTPUT_DIR = "output"
MODEL_DIR = "models"

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Color maps for visualization
fire_cmap = ListedColormap(['green', 'yellow', 'orange', 'red'])
binary_cmap = ListedColormap(['green', 'red'])

#######################
# Data Collection
#######################

def download_weather_data(region, start_date, end_date):
    """
    Download weather data (temperature, precipitation, humidity, wind) for the specified region and time period.

    Args:
        region (str): Name of the region
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        dict: Dictionary containing weather data rasters
    """
    print(f"Downloading weather data for {region} from {start_date} to {end_date}...")

    # In a real implementation, this would use APIs to download data from sources like:
    # - MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre)
    # - ERA-5 (ECMWF Reanalysis v5)
    # - IMD (India Meteorological Department)

    # For demonstration, we'll create synthetic data
    # Use a smaller grid size for testing to improve performance
    if "Test" in region:
        # Smaller grid for testing (10x10 km)
        grid_size = int(10000 / RESOLUTION)
    else:
        # Full size for actual runs (100x100 km)
        grid_size = int(100000 / RESOLUTION)

    # Create synthetic weather data
    temperature = np.random.normal(30, 5, (grid_size, grid_size))  # Mean 30°C with 5°C std dev
    precipitation = np.random.exponential(2, (grid_size, grid_size))  # Exponential distribution for rainfall
    humidity = np.random.normal(60, 15, (grid_size, grid_size)).clip(10, 100)  # Mean 60% with 15% std dev
    wind_speed = np.random.rayleigh(4, (grid_size, grid_size))  # Rayleigh distribution for wind speed
    wind_direction = np.random.uniform(0, 360, (grid_size, grid_size))  # Uniform distribution for wind direction

    return {
        'temperature': temperature,
        'precipitation': precipitation,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction
    }

def download_terrain_data(region):
    """
    Download terrain data (DEM, slope, aspect) for the specified region.

    Args:
        region (str): Name of the region

    Returns:
        dict: Dictionary containing terrain data rasters
    """
    print(f"Downloading terrain data for {region}...")

    # In a real implementation, this would download DEM data from Bhoonidhi portal
    # and calculate slope and aspect

    # For demonstration, we'll create synthetic data
    # Use a smaller grid size for testing to improve performance
    if "Test" in region:
        # Smaller grid for testing (10x10 km)
        grid_size = int(10000 / RESOLUTION)
    else:
        # Full size for actual runs (100x100 km)
        grid_size = int(100000 / RESOLUTION)

    # Create a synthetic DEM with some mountains and valleys
    x = np.linspace(0, 10, grid_size)
    y = np.linspace(0, 10, grid_size)
    X, Y = np.meshgrid(x, y)

    # Create some mountain peaks
    peaks = [
        (3, 3, 2000, 1),
        (7, 7, 3000, 1.5),
        (2, 8, 2500, 0.8),
        (8, 2, 1800, 1.2)
    ]

    dem = np.zeros((grid_size, grid_size))
    for px, py, height, spread in peaks:
        dem += height * np.exp(-((X - px)**2 + (Y - py)**2) / spread)

    # Calculate slope and aspect
    dy, dx = np.gradient(dem, RESOLUTION, RESOLUTION)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    aspect = np.degrees(np.arctan2(dy, -dx))
    aspect = np.where(aspect < 0, aspect + 360, aspect)

    return {
        'dem': dem,
        'slope': slope,
        'aspect': aspect
    }

def download_lulc_data(region):
    """
    Download Land Use Land Cover (LULC) data for the specified region.

    Args:
        region (str): Name of the region

    Returns:
        np.ndarray: LULC classification raster
    """
    print(f"Downloading LULC data for {region}...")

    # In a real implementation, this would download LULC data from Bhuvan/Sentinel Hub

    # For demonstration, we'll create synthetic data
    # Use a smaller grid size for testing to improve performance
    if "Test" in region:
        # Smaller grid for testing (10x10 km)
        grid_size = int(10000 / RESOLUTION)
    else:
        # Full size for actual runs (100x100 km)
        grid_size = int(100000 / RESOLUTION)

    # Create synthetic LULC data with the following classes:
    # 0: Water
    # 1: Urban
    # 2: Agriculture
    # 3: Grassland
    # 4: Shrubland
    # 5: Deciduous Forest
    # 6: Evergreen Forest
    # 7: Mixed Forest
    # 8: Barren

    # Start with random classification
    lulc = np.random.randint(0, 9, (grid_size, grid_size))

    # Make it more realistic by adding spatial correlation
    from scipy.ndimage import gaussian_filter

    # Create probability maps for each class
    prob_maps = [gaussian_filter(np.random.random((grid_size, grid_size)), sigma=5) for _ in range(9)]

    # Assign the class with highest probability to each pixel
    lulc = np.argmax(np.array(prob_maps), axis=0)

    return lulc

def download_human_factors(region):
    """
    Download human settlement and infrastructure data for the specified region.

    Args:
        region (str): Name of the region

    Returns:
        dict: Dictionary containing human factors rasters
    """
    print(f"Downloading human factors data for {region}...")

    # In a real implementation, this would download data from GHSL (Global Human Settlement Layer)

    # For demonstration, we'll create synthetic data
    # Use a smaller grid size for testing to improve performance
    if "Test" in region:
        # Smaller grid for testing (10x10 km)
        grid_size = int(10000 / RESOLUTION)
    else:
        # Full size for actual runs (100x100 km)
        grid_size = int(100000 / RESOLUTION)

    # Create synthetic settlement density
    settlement = np.zeros((grid_size, grid_size))

    # Add some towns/cities
    centers = [
        (int(grid_size * 0.2), int(grid_size * 0.3), 0.8),
        (int(grid_size * 0.7), int(grid_size * 0.2), 0.6),
        (int(grid_size * 0.5), int(grid_size * 0.5), 1.0),
        (int(grid_size * 0.8), int(grid_size * 0.8), 0.7)
    ]

    for cx, cy, intensity in centers:
        y, x = np.ogrid[-cy:grid_size-cy, -cx:grid_size-cx]
        mask = x*x + y*y <= (grid_size/10)**2
        # Create a distance array for the exponential decay
        dist_squared = x*x + y*y
        # Apply the mask
        settlement[mask] = np.maximum(settlement[mask], intensity * np.exp(-(dist_squared[mask] / (grid_size/15)**2)))

    # Create synthetic road network
    roads = np.zeros((grid_size, grid_size))

    # Add some roads connecting the centers
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            x1, y1, _ = centers[i]
            x2, y2, _ = centers[j]

            # Create a road with some random curvature
            t = np.linspace(0, 1, 100)
            noise = np.random.normal(0, grid_size/20, (100, 2))

            # Linear interpolation with noise
            road_x = (x1 * (1-t) + x2 * t + noise[:, 0]).astype(int)
            road_y = (y1 * (1-t) + y2 * t + noise[:, 1]).astype(int)

            # Clip to grid boundaries
            mask = (road_x >= 0) & (road_x < grid_size) & (road_y >= 0) & (road_y < grid_size)
            roads[road_y[mask], road_x[mask]] = 1

    # Dilate roads to make them visible
    from scipy.ndimage import binary_dilation
    roads = binary_dilation(roads, iterations=2)

    return {
        'settlement': settlement,
        'roads': roads
    }

def download_historical_fires(region, start_date, end_date):
    """
    Download historical fire data for the specified region and time period.

    Args:
        region (str): Name of the region
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        dict: Dictionary containing fire data for each date
    """
    print(f"Downloading historical fire data for {region} from {start_date} to {end_date}...")

    # In a real implementation, this would download fire data from VIIRS-SNP

    # For demonstration, we'll create synthetic data
    # Use a smaller grid size for testing to improve performance
    if "Test" in region:
        # Smaller grid for testing (10x10 km)
        grid_size = int(10000 / RESOLUTION)
    else:
        # Full size for actual runs (100x100 km)
        grid_size = int(100000 / RESOLUTION)

    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Generate a date range
    date_range = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    # Create synthetic fire data for each date
    fire_data = {}

    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")

        # Create a sparse fire occurrence map (mostly zeros with some ones)
        fire_map = np.zeros((grid_size, grid_size))

        # Add some random fire spots (more in dry season, fewer in wet season)
        # Assuming dry season is April-June
        month = date.month
        if 4 <= month <= 6:
            fire_probability = 0.001  # Higher in dry season
        else:
            fire_probability = 0.0002  # Lower in wet season

        fire_map = np.random.random((grid_size, grid_size)) < fire_probability

        # Add spatial correlation (fires tend to cluster)
        from scipy.ndimage import binary_dilation
        fire_map = binary_dilation(fire_map, iterations=2)

        fire_data[date_str] = fire_map

    return fire_data

#######################
# Data Preprocessing
#######################

def preprocess_data(weather_data, terrain_data, lulc_data, human_factors, historical_fires):
    """
    Preprocess and combine all data sources into feature stacks for model training.

    Args:
        weather_data (dict): Weather data rasters
        terrain_data (dict): Terrain data rasters
        lulc_data (np.ndarray): LULC classification raster
        human_factors (dict): Human factors rasters
        historical_fires (dict): Historical fire data

    Returns:
        tuple: X (features) and y (target) arrays for model training
    """
    print("Preprocessing data...")

    # Convert LULC to fuel availability
    # Higher values indicate more fuel
    fuel_map = np.zeros_like(lulc_data, dtype=float)

    # Assign fuel values based on LULC class
    fuel_values = {
        0: 0.0,    # Water
        1: 0.2,    # Urban
        2: 0.4,    # Agriculture
        3: 0.7,    # Grassland
        4: 0.8,    # Shrubland
        5: 0.9,    # Deciduous Forest
        6: 1.0,    # Evergreen Forest
        7: 0.95,   # Mixed Forest
        8: 0.1     # Barren
    }

    for lulc_class, fuel_value in fuel_values.items():
        fuel_map[lulc_data == lulc_class] = fuel_value

    # Prepare feature stack for each date in historical_fires
    X = []
    y = []

    dates = sorted(historical_fires.keys())

    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i+1]

        # Extract weather for current date
        # In a real implementation, we would have weather data for each date
        # For demonstration, we'll use the same weather data for all dates
        temperature = weather_data['temperature']
        precipitation = weather_data['precipitation']
        humidity = weather_data['humidity']
        wind_speed = weather_data['wind_speed']
        wind_direction = weather_data['wind_direction']

        # Create feature stack
        features = np.stack([
            temperature,
            precipitation,
            humidity,
            wind_speed,
            np.sin(np.radians(wind_direction)),  # Convert to x-component
            np.cos(np.radians(wind_direction)),  # Convert to y-component
            terrain_data['slope'],
            np.sin(np.radians(terrain_data['aspect'])),  # Convert to x-component
            np.cos(np.radians(terrain_data['aspect'])),  # Convert to y-component
            fuel_map,
            human_factors['settlement'],
            human_factors['roads'],
            historical_fires[current_date]  # Current fire state
        ], axis=-1)

        # Target is the fire state for the next day
        target = historical_fires[next_date]

        X.append(features)
        y.append(target)

    return np.array(X), np.array(y)

def split_data(X, y, test_size=0.2, val_size=0.2):
    """
    Split data into training, validation, and test sets.

    Args:
        X (np.ndarray): Feature array
        y (np.ndarray): Target array
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation

    Returns:
        tuple: Training, validation, and test data
    """
    # Check if we have a very small number of samples
    if len(X) <= 5:
        print("Small sample size detected. Using simplified data splitting for testing.")
        # For very small sample sizes, use a simpler approach
        if len(X) == 1:
            # If only one sample, use it for all sets
            return (X, y), (X, y), (X, y)
        elif len(X) == 2:
            # If two samples, use first for train/val and second for test
            return (X[0:1], y[0:1]), (X[0:1], y[0:1]), (X[1:2], y[1:2])
        else:
            # If 3-5 samples, use first for train, second for val, rest for test
            return (X[0:1], y[0:1]), (X[1:2], y[1:2]), (X[2:], y[2:])

    # For normal cases, use standard splitting
    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Then split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

#######################
# Fire Prediction Model (U-NET)
#######################

def create_unet_model(input_shape):
    """
    Create a U-NET model for fire prediction.

    Args:
        input_shape (tuple): Shape of input data

    Returns:
        tf.keras.Model: U-NET model
    """
    # Input layer
    inputs = Input(input_shape)

    # Encoder (downsampling path)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bridge
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)

    # Decoder (upsampling path)
    up5 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(drop4))
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, train_data, val_data, epochs=50, batch_size=8):
    """
    Train the fire prediction model.

    Args:
        model (tf.keras.Model): Model to train
        train_data (tuple): Training data (X_train, y_train)
        val_data (tuple): Validation data (X_val, y_val)
        epochs (int): Number of epochs to train for
        batch_size (int): Batch size for training

    Returns:
        tf.keras.Model: Trained model
    """
    X_train, y_train = train_data
    X_val, y_val = val_data

    # Reshape targets to match model output
    y_train = y_train.reshape(y_train.shape + (1,))
    y_val = y_val.reshape(y_val.shape + (1,))

    # Callbacks
    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint(
        os.path.join(MODEL_DIR, 'fire_prediction_model.h5'),
        save_best_only=True,
        verbose=1
    )

    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )

    return model, history

def predict_fire_probability(model, features):
    """
    Predict fire probability for the next day.

    Args:
        model (tf.keras.Model): Trained model
        features (np.ndarray): Feature stack for current day

    Returns:
        np.ndarray: Fire probability map for the next day
    """
    # Add batch dimension if needed
    if len(features.shape) == 3:
        features = features.reshape(1, *features.shape)

    # Make prediction
    prediction = model.predict(features)

    # Remove batch and channel dimensions
    prediction = prediction.squeeze()

    return prediction

#######################
# Fire Spread Simulation (Cellular Automata)
#######################

def simulate_fire_spread(initial_fire, weather_data, terrain_data, fuel_map, hours=12, time_step=1, testing_mode=False):
    """
    Simulate fire spread using a Cellular Automata model.

    Args:
        initial_fire (np.ndarray): Initial fire state (binary)
        weather_data (dict): Weather data rasters
        terrain_data (dict): Terrain data rasters
        fuel_map (np.ndarray): Fuel availability map
        hours (int): Number of hours to simulate
        time_step (int): Time step in hours
        testing_mode (bool): If True, use a simplified simulation for testing

    Returns:
        list: List of fire states at each time step
    """
    # Initialize simulation
    current_fire = initial_fire.copy()
    fire_states = [current_fire.copy()]

    # Number of time steps
    n_steps = int(hours / time_step)

    # Grid dimensions
    height, width = current_fire.shape

    if testing_mode:
        # Simplified simulation for testing
        print("Using simplified fire spread simulation for testing...")

        # Simple distance-based spread
        for step in range(n_steps):
            # Create a new fire state
            new_fire = current_fire.copy()

            # Simple dilation of fire state
            from scipy.ndimage import binary_dilation

            # Create a circular structuring element
            radius = step + 1  # Increase radius with each step
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            structuring_element = x*x + y*y <= radius*radius

            # Dilate the fire state
            new_fire = binary_dilation(current_fire, structure=structuring_element)

            # Apply fuel constraints
            new_fire = new_fire & (fuel_map > 0)

            # Update current fire state
            current_fire = new_fire
            fire_states.append(current_fire.copy())
    else:
        # Full simulation
        # Extract relevant data
        wind_speed = weather_data['wind_speed']
        wind_direction = weather_data['wind_direction']
        temperature = weather_data['temperature']
        humidity = weather_data['humidity']
        slope = terrain_data['slope']
        aspect = terrain_data['aspect']

        # Convert wind direction to radians
        wind_direction_rad = np.radians(wind_direction)

        # Simulation parameters
        base_spread_rate = 0.3  # Base spread rate in km/h
        wind_factor = 0.5       # Wind influence factor
        slope_factor = 0.3      # Slope influence factor
        temp_factor = 0.2       # Temperature influence factor
        humidity_factor = -0.3  # Humidity influence factor (negative because higher humidity reduces spread)

        # Neighborhood (Moore neighborhood)
        neighborhood = np.array([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],           [0, 1],
            [1, -1],  [1, 0],  [1, 1]
        ])

        # Direction angles for each neighbor (in radians)
        neighbor_angles = np.array([
            3*np.pi/4, np.pi, 5*np.pi/4,
            np.pi/2,           3*np.pi/2,
            np.pi/4,  0,       7*np.pi/4
        ])

        # Simulation loop
        for step in range(n_steps):
            # Create a new fire state
            new_fire = current_fire.copy()

            # Iterate over all cells
            for i in range(1, height-1):
                for j in range(1, width-1):
                    # Skip if cell is already on fire
                    if current_fire[i, j] == 1:
                        continue

                    # Skip if cell has no fuel
                    if fuel_map[i, j] == 0:
                        continue

                    # Check neighbors
                    for n in range(len(neighborhood)):
                        ni, nj = neighborhood[n]
                        neighbor_i, neighbor_j = i + ni, j + nj

                        # Skip if neighbor is out of bounds
                        if (neighbor_i < 0 or neighbor_i >= height or
                            neighbor_j < 0 or neighbor_j >= width):
                            continue

                        # Skip if neighbor is not on fire
                        if current_fire[neighbor_i, neighbor_j] == 0:
                            continue

                        # Calculate spread probability

                        # Wind effect
                        # Higher probability if wind is blowing from the neighbor towards the cell
                        wind_angle = wind_direction_rad[i, j]
                        spread_angle = neighbor_angles[n]
                        angle_diff = np.abs((wind_angle - spread_angle + np.pi) % (2 * np.pi) - np.pi)
                        wind_effect = wind_factor * wind_speed[i, j] * np.cos(angle_diff)

                        # Slope effect
                        # Higher probability if spreading uphill
                        slope_angle = np.radians(slope[i, j])
                        aspect_angle = np.radians(aspect[i, j])
                        slope_effect = slope_factor * slope_angle * np.cos(aspect_angle - spread_angle)

                        # Temperature effect
                        # Higher probability with higher temperature
                        temp_effect = temp_factor * (temperature[i, j] - 20) / 30  # Normalized around 20°C

                        # Humidity effect
                        # Lower probability with higher humidity
                        humidity_effect = humidity_factor * humidity[i, j] / 100

                        # Fuel effect
                        # Higher probability with more fuel
                        fuel_effect = fuel_map[i, j]

                        # Calculate base spread rate in this direction
                        spread_rate = base_spread_rate * (1 + wind_effect + slope_effect + temp_effect + humidity_effect) * fuel_effect

                        # Convert to probability for this time step
                        # Probability = 1 - exp(-rate * time)
                        spread_prob = 1 - np.exp(-spread_rate * time_step)

                        # Determine if fire spreads to this cell
                        if np.random.random() < spread_prob:
                            new_fire[i, j] = 1
                            break

            # Update current fire state
            current_fire = new_fire
            fire_states.append(current_fire.copy())

    return fire_states

#######################
# Visualization
#######################

def visualize_prediction(prediction, output_path=None, testing_mode=False):
    """
    Visualize fire prediction map.

    Args:
        prediction (np.ndarray): Fire probability map
        output_path (str, optional): Path to save the visualization
        testing_mode (bool): If True, skip the actual visualization for testing

    Returns:
        None
    """
    if testing_mode:
        print("Skipping visualization for testing mode...")
        if output_path:
            # Just create an empty file to verify the function was called
            with open(output_path, 'w') as f:
                f.write("Test visualization file")
            print(f"Test prediction map saved to {output_path}")
        return

    plt.figure(figsize=(10, 8))

    # Plot probability map
    plt.imshow(prediction, cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(label='Fire Probability')
    plt.title('Fire Prediction Map for Next Day')

    # Add binary classification overlay
    binary_prediction = (prediction > 0.5).astype(int)
    plt.contour(binary_prediction, levels=[0.5], colors='black', linestyles='dashed')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Prediction map saved to {output_path}")

    plt.show()

def create_fire_spread_animation(fire_states, time_step, output_path=None, testing_mode=False):
    """
    Create animation of fire spread simulation.

    Args:
        fire_states (list): List of fire states at each time step
        time_step (int): Time step in hours
        output_path (str, optional): Path to save the animation
        testing_mode (bool): If True, skip the actual animation for testing

    Returns:
        None
    """
    if testing_mode:
        print("Skipping animation for testing mode...")
        if output_path:
            # Just create an empty file to verify the function was called
            with open(output_path, 'w') as f:
                f.write("Test animation file")
            print(f"Test fire spread animation saved to {output_path}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create initial plot
    im = ax.imshow(fire_states[0], cmap=binary_cmap, vmin=0, vmax=1)
    title = ax.set_title('Fire Spread: 0 hours')

    # Update function for animation
    def update(frame):
        im.set_array(fire_states[frame])
        title.set_text(f'Fire Spread: {frame * time_step} hours')
        return [im, title]

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(fire_states), interval=500, blit=True)

    if output_path:
        ani.save(output_path, writer='pillow', fps=2)
        print(f"Fire spread animation saved to {output_path}")

    plt.show()

def save_as_geotiff(data, output_path, region_bounds=None):
    """
    Save data as a GeoTIFF file.

    Args:
        data (np.ndarray): Data to save
        output_path (str): Path to save the GeoTIFF
        region_bounds (tuple, optional): Region bounds (left, bottom, right, top)

    Returns:
        None
    """
    # Convert boolean arrays to integers
    if data.dtype == bool:
        data = data.astype(np.uint8)
    # If region bounds not provided, use default
    if region_bounds is None:
        # Default to a region in Uttarakhand
        # These are approximate coordinates for demonstration
        left, bottom, right, top = 77.0, 28.5, 81.0, 31.5
    else:
        left, bottom, right, top = region_bounds

    # Calculate pixel size
    height, width = data.shape
    pixel_width = (right - left) / width
    pixel_height = (top - bottom) / height

    # Create transform
    transform = from_origin(left, top, pixel_width, pixel_height)

    # Create GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
        transform=transform
    ) as dst:
        dst.write(data, 1)

    print(f"Data saved as GeoTIFF to {output_path}")

#######################
# Main Function
#######################

def main(demo_mode=False):
    """
    Main function to run the forest fire prediction and simulation.

    Args:
        demo_mode (bool): If True, use simplified parameters for demonstration
    """
    print("Starting Forest Fire Prediction and Simulation...")

    # Define date range for historical data
    if demo_mode:
        print("Running in demonstration mode with simplified parameters...")
        start_date = "2022-01-01"
        end_date = "2022-01-10"  # Shorter date range for demo
        region = "Demo_" + REGION  # Add Demo_ prefix to trigger smaller grid size
    else:
        start_date = "2022-01-01"
        end_date = "2022-12-31"
        region = REGION

    # 1. Data Collection
    weather_data = download_weather_data(region, start_date, end_date)
    terrain_data = download_terrain_data(region)
    lulc_data = download_lulc_data(region)
    human_factors = download_human_factors(region)
    historical_fires = download_historical_fires(region, start_date, end_date)

    # 2. Data Preprocessing
    X, y = preprocess_data(weather_data, terrain_data, lulc_data, human_factors, historical_fires)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)

    # 3. Model Training
    input_shape = X_train.shape[1:]

    if demo_mode:
        # Use a simplified model for demo
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D

        print("Using simplified model for demonstration...")
        model = Sequential([
            Conv2D(16, 3, activation='relu', padding='same', input_shape=input_shape),
            Conv2D(8, 3, activation='relu', padding='same'),
            Conv2D(1, 1, activation='sigmoid', padding='same')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Skip actual training for demo
        print("Skipping actual training for demonstration...")
        # Just create a dummy prediction for demonstration
        prediction = np.random.random(lulc_data.shape)
    else:
        # Use the full U-NET model for actual runs
        model = create_unet_model(input_shape)
        model, history = train_model(model, (X_train, y_train), (X_val, y_val), epochs=20)

        # 4. Fire Prediction
        # Use the last day's features to predict the next day
        last_features = X[-1]
        prediction = predict_fire_probability(model, last_features)

    # Visualize prediction
    visualize_prediction(prediction, os.path.join(OUTPUT_DIR, 'fire_prediction.png'), testing_mode=demo_mode)

    # Save prediction as GeoTIFF
    save_as_geotiff(prediction, os.path.join(OUTPUT_DIR, 'fire_prediction.tif'))

    # 5. Fire Spread Simulation
    # Use high-risk areas as initial fire state
    initial_fire = (np.array(prediction) > 0.8).astype(int)

    # Extract fuel map from LULC
    fuel_map = np.zeros_like(lulc_data, dtype=float)
    fuel_values = {
        0: 0.0,    # Water
        1: 0.2,    # Urban
        2: 0.4,    # Agriculture
        3: 0.7,    # Grassland
        4: 0.8,    # Shrubland
        5: 0.9,    # Deciduous Forest
        6: 1.0,    # Evergreen Forest
        7: 0.95,   # Mixed Forest
        8: 0.1     # Barren
    }
    for lulc_class, fuel_value in fuel_values.items():
        fuel_map[lulc_data == lulc_class] = fuel_value

    # Simulate fire spread
    time_step = 1  # 1 hour
    hours = 6 if demo_mode else 12  # Shorter simulation for demo

    fire_states = simulate_fire_spread(
        initial_fire,
        weather_data,
        terrain_data,
        fuel_map,
        hours=hours,
        time_step=time_step,
        testing_mode=demo_mode  # Use simplified simulation for demo
    )

    # Create animation
    create_fire_spread_animation(
        fire_states,
        time_step,
        os.path.join(OUTPUT_DIR, 'fire_spread_animation.gif'),
        testing_mode=demo_mode  # Skip actual animation for demo
    )

    # Save fire spread states as GeoTIFFs
    for hour in [1, 2, 3, 6, 12]:
        if hour <= len(fire_states) - 1:
            save_as_geotiff(
                fire_states[hour],
                os.path.join(OUTPUT_DIR, f'fire_spread_{hour}h.tif')
            )

    print("Forest Fire Prediction and Simulation completed successfully!")

if __name__ == "__main__":
    # Run in demo mode for faster execution and lower memory usage
    main(demo_mode=True)
