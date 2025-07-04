
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
# import geopandas as gpd # Unused
# from datetime import datetime, timedelta # Unused in this file (used in synthetic_data_generator)
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

# Import synthetic data generator
import synthetic_data_generator as sdg

#######################
# Data Collection
#######################

# Flag to control data source: True for synthetic, False for actual (placeholder)
USE_SYNTHETIC_DATA = True

def download_weather_data(region, start_date, end_date, use_synthetic=USE_SYNTHETIC_DATA):
    """
    Download weather data (temperature, precipitation, humidity, wind) for the specified region and time period.
    If use_synthetic is True, generates synthetic data. Otherwise, this is a placeholder for actual data download.

    Args:
        region (str): Name of the region
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        use_synthetic (bool): If True, generate synthetic data.

    Returns:
        dict: Dictionary containing weather data rasters
    """
    print(f"Requesting weather data for {region} from {start_date} to {end_date}...")
    if use_synthetic:
        print("Generating synthetic weather data...")
        grid_size = sdg.get_grid_size(region, RESOLUTION)
        return sdg.generate_synthetic_weather_data(grid_size, start_date, end_date)
    else:
        # Placeholder for actual data download logic
        # This would use APIs to download data from sources like:
        # - MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre)
        # - ERA-5 (ECMWF Reanalysis v5)
        # - IMD (India Meteorological Department)
        print("Actual weather data download not implemented. Returning empty dictionary.")
        return {}


def download_terrain_data(region, use_synthetic=USE_SYNTHETIC_DATA):
    """
    Download terrain data (DEM, slope, aspect) for the specified region.
    If use_synthetic is True, generates synthetic data. Otherwise, this is a placeholder for actual data download.

    Args:
        region (str): Name of the region
        use_synthetic (bool): If True, generate synthetic data.

    Returns:
        dict: Dictionary containing terrain data rasters
    """
    print(f"Requesting terrain data for {region}...")
    if use_synthetic:
        print("Generating synthetic terrain data...")
        grid_size = sdg.get_grid_size(region, RESOLUTION)
        return sdg.generate_synthetic_terrain_data(grid_size)
    else:
        # Placeholder for actual data download logic
        # This would download DEM data from Bhoonidhi portal and calculate slope and aspect
        print("Actual terrain data download not implemented. Returning empty dictionary.")
        return {}


def download_lulc_data(region, use_synthetic=USE_SYNTHETIC_DATA):
    """
    Download Land Use Land Cover (LULC) data for the specified region.
    If use_synthetic is True, generates synthetic data. Otherwise, this is a placeholder for actual data download.

    Args:
        region (str): Name of the region
        use_synthetic (bool): If True, generate synthetic data.

    Returns:
        np.ndarray: LULC classification raster or None if not implemented
    """
    print(f"Requesting LULC data for {region}...")
    if use_synthetic:
        print("Generating synthetic LULC data...")
        grid_size = sdg.get_grid_size(region, RESOLUTION)
        return sdg.generate_synthetic_lulc_data(grid_size)
    else:
        # Placeholder for actual data download logic
        # This would download LULC data from Bhuvan/Sentinel Hub
        print("Actual LULC data download not implemented. Returning None.")
        return None


def download_human_factors(region, use_synthetic=USE_SYNTHETIC_DATA):
    """
    Download human settlement and infrastructure data for the specified region.
    If use_synthetic is True, generates synthetic data. Otherwise, this is a placeholder for actual data download.

    Args:
        region (str): Name of the region
        use_synthetic (bool): If True, generate synthetic data.

    Returns:
        dict: Dictionary containing human factors rasters or empty dict if not implemented
    """
    print(f"Requesting human factors data for {region}...")
    if use_synthetic:
        print("Generating synthetic human factors data...")
        grid_size = sdg.get_grid_size(region, RESOLUTION)
        return sdg.generate_synthetic_human_factors(grid_size)
    else:
        # Placeholder for actual data download logic
        # This would download data from GHSL (Global Human Settlement Layer)
        print("Actual human factors data download not implemented. Returning empty dictionary.")
        return {}


def download_historical_fires(region, start_date, end_date, use_synthetic=USE_SYNTHETIC_DATA):
    """
    Download historical fire data for the specified region and time period.
    If use_synthetic is True, generates synthetic data. Otherwise, this is a placeholder for actual data download.

    Args:
        region (str): Name of the region
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        use_synthetic (bool): If True, generate synthetic data.

    Returns:
        dict: Dictionary containing fire data for each date or empty dict if not implemented
    """
    print(f"Requesting historical fire data for {region} from {start_date} to {end_date}...")
    if use_synthetic:
        print("Generating synthetic historical fire data...")
        grid_size = sdg.get_grid_size(region, RESOLUTION)
        return sdg.generate_synthetic_historical_fires(grid_size, start_date, end_date)
    else:
        # Placeholder for actual data download logic
        # This would download fire data from VIIRS-SNP
        print("Actual historical fire data download not implemented. Returning empty dictionary.")
        return {}

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
    fuel_map = np.zeros_like(lulc_data, dtype=float)
    fuel_values = {
        0: 0.0, 1: 0.2, 2: 0.4, 3: 0.7, 4: 0.8,
        5: 0.9, 6: 1.0, 7: 0.95, 8: 0.1  # Barren
    }
    for lulc_class, fuel_value in fuel_values.items():
        fuel_map[lulc_data == lulc_class] = fuel_value

    # Pre-calculate components of wind and aspect
    # Ensure weather_data and terrain_data are not empty (can happen if actual download fails)
    if not weather_data or not terrain_data:
        print("Error: Weather or terrain data is missing. Cannot preprocess.")
        # Return empty arrays with expected number of dimensions
        height, width = fuel_map.shape
        # Assuming a default number of features if data is missing, this might need adjustment
        # For now, let's estimate based on typical structure.
        # num_static_features (12) + current_fire_state (1) = 13
        num_features = 13
        return np.empty((0, height, width, num_features)), np.empty((0, height, width))

    wind_dir_rad = np.radians(weather_data['wind_direction'])
    aspect_rad = np.radians(terrain_data['aspect'])

    sin_wind_direction = np.sin(wind_dir_rad)
    cos_wind_direction = np.cos(wind_dir_rad)
    sin_aspect = np.sin(aspect_rad)
    cos_aspect = np.cos(aspect_rad)

    # Static features that don't change with date (assuming weather data is static as per current implementation)
    static_feature_layers = [
        weather_data['temperature'],
        weather_data['precipitation'],
        weather_data['humidity'],
        weather_data['wind_speed'],
        sin_wind_direction,
        cos_wind_direction,
        terrain_data['slope'],
        sin_aspect,
        cos_aspect,
        fuel_map,
        human_factors['settlement'],
        human_factors['roads'],
    ]
    # Stack static features once. Resulting shape: (height, width, num_static_features)
    static_features_stack = np.stack(static_feature_layers, axis=-1)

    X_list = []
    y_list = []

    if not historical_fires:
        print("Warning: Historical fire data is empty.")
        dates = []
    else:
        dates = sorted(historical_fires.keys())

    if not dates or len(dates) < 2:
        print("Warning: Not enough historical fire data to create training samples (need at least 2 days).")
        num_features = static_features_stack.shape[-1] + 1 # +1 for current fire state
        height, width = fuel_map.shape
        return np.empty((0, height, width, num_features)), np.empty((0, height, width))

    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i+1]

        current_fire_state = historical_fires[current_date].astype(float)
        current_fire_state_reshaped = current_fire_state[..., np.newaxis]

        features_for_date = np.concatenate((static_features_stack, current_fire_state_reshaped), axis=-1)

        X_list.append(features_for_date)
        y_list.append(historical_fires[next_date].astype(float))

    if not X_list: # Should be covered by earlier checks, but as a safeguard
        num_features = static_features_stack.shape[-1] + 1
        height, width = fuel_map.shape
        return np.empty((0, height, width, num_features)), np.empty((0, height, width))

    return np.array(X_list), np.array(y_list)

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
import numba

# Constants for Numba-optimized fire spread
NEIGHBORHOOD_OFFSETS = np.array([
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1],           [0, 1], # Moore neighborhood
    [1, -1],  [1, 0],  [1, 1]
], dtype=np.intp)

# Corrected mapping for NEIGHBOR_ANGLES_RAD to match NEIGHBORHOOD_OFFSETS standard indexing:
# Offset (-1,-1) NW: angle should be 3*pi/4
# Offset (-1, 0) N : angle should be pi/2
# Offset (-1, 1) NE: angle should be pi/4
# Offset ( 0,-1) W : angle should be pi
# Offset ( 0, 1) E : angle should be 0
# Offset ( 1,-1) SW: angle should be 5*pi/4
# Offset ( 1, 0) S : angle should be 3*pi/2
# Offset ( 1, 1) SE: angle should be 7*pi/4
# Offset (-1,-1) NW: angle should be 3*pi/4
# Offset (-1, 0) N : angle should be pi/2
# Offset (-1, 1) NE: angle should be pi/4
# Offset ( 0,-1) W : angle should be pi
# Offset ( 0, 1) E : angle should be 0
# Offset ( 1,-1) SW: angle should be 5*pi/4
# Offset ( 1, 0) S : angle should be 3*pi/2
# Offset ( 1, 1) SE: angle should be 7*pi/4
NEIGHBOR_ANGLES_RAD_CORRECTED = np.array([
    3*np.pi/4, np.pi/2, np.pi/4,      # For (-1,-1), (-1,0), (-1,1)
    np.pi,              0,            # For (0,-1), (0,1)
    5*np.pi/4, 3*np.pi/2, 7*np.pi/4   # For (1,-1), (1,0), (1,1)
], dtype=np.float64)


@numba.jit(nopython=True, fastmath=True)
def _simulate_fire_spread_step_numba(current_fire, fuel_map,
                                     wind_speed_map, wind_direction_rad_map,
                                     temperature_map, humidity_map,
                                     slope_deg_map, aspect_rad_map,
                                     base_spread_rate, wind_factor, slope_factor,
                                     temp_factor, humidity_factor, time_step,
                                     neighborhood_offsets, neighbor_angles_rad):
    height, width = current_fire.shape
    new_fire = current_fire.copy()

    for i in range(1, height - 1): # Iterate excluding borders
        for j in range(1, width - 1): # Iterate excluding borders
            if current_fire[i, j] == 1: # Already burning
                continue
            if fuel_map[i, j] == 0: # No fuel
                continue

            # Cell-specific properties for cell (i,j)
            fuel_at_ij = fuel_map[i, j]
            # Normalize temperature around 20°C, range roughly -1 to 1 for 0-50°C
            temp_effect_at_ij = temp_factor * (temperature_map[i, j] - 20.0) / 30.0
            # Normalize humidity 0-100 to 0-1
            humidity_effect_at_ij = humidity_factor * humidity_map[i, j] / 100.0
            wind_speed_at_ij = wind_speed_map[i, j]
            wind_angle_at_ij = wind_direction_rad_map[i, j] # radians

            slope_deg_at_ij = slope_deg_map[i,j] # degrees
            slope_rad_at_ij = np.radians(slope_deg_at_ij) # Numba np.radians for float
            aspect_rad_at_ij = aspect_rad_map[i,j] # radians

            # Check neighbors
            for n_idx in range(neighborhood_offsets.shape[0]):
                ni, nj = neighborhood_offsets[n_idx, 0], neighborhood_offsets[n_idx, 1]
                neighbor_i, neighbor_j = i + ni, j + nj

                # No need to check neighbor bounds, (i,j) loop range ensures neighbors are valid.

                if current_fire[neighbor_i, neighbor_j] == 1: # If neighbor is burning
                    spread_angle = neighbor_angles_rad[n_idx] # radians; angle from (i,j) TO neighbor

                    # Wind effect: positive if wind blows from neighbor towards (i,j)
                    # Angle from neighbor to (i,j) is spread_angle + pi
                    # Effective wind direction relative to spread: wind_angle_at_ij - (spread_angle + pi)
                    # Original: angle_diff = np.abs((wind_angle_at_ij - spread_angle + np.pi) % (2 * np.pi) - np.pi)
                    # This calculates smallest angle between wind_angle_at_ij and spread_angle.
                    # A positive wind_effect means wind is aligned with the spread direction from neighbor.
                    # spread_angle is direction from cell (i,j) *to* neighbor.
                    # We need direction *from* neighbor *to* cell (i,j). This is `spread_angle` if defined from neighbor,
                    # or `(neighbor_angles_rad[n_idx] + np.pi) % (2*np.pi)` if `neighbor_angles_rad` is direction from cell to neighbor.
                    # Let's assume neighbor_angles_rad[n_idx] is direction FROM (i,j) TO neighbor.
                    # So, direction of spread (fire coming from neighbor) is (spread_angle + pi)
                    direction_from_neighbor_to_cell = (spread_angle + np.pi) % (2.0 * np.pi)
                    angle_diff_wind = np.abs((wind_angle_at_ij - direction_from_neighbor_to_cell + np.pi) % (2 * np.pi) - np.pi)
                    wind_effect = wind_factor * wind_speed_at_ij * np.cos(angle_diff_wind)

                    # Slope effect: positive if spreading uphill (from neighbor to (i,j))
                    # spread_angle is direction from (i,j) to neighbor.
                    # Slope effect on cell (i,j) when fire comes from neighbor along (spread_angle + pi) direction.
                    # Original: slope_effect = slope_factor * slope_rad_at_ij * np.cos(aspect_rad_at_ij - spread_angle)
                    # This formula means: if aspect is aligned with spread_angle (i.e. slope is "facing" the neighbor),
                    # then cos is positive, effect is positive. This seems to be "slope receptivity to fire from that direction".
                    # If spreading uphill TO (i,j), then (i,j) is higher than neighbor.
                    # The original formula seems to calculate how slope at (i,j) affects fire coming from a certain direction.
                    slope_effect = slope_factor * slope_rad_at_ij * np.cos(aspect_rad_at_ij - direction_from_neighbor_to_cell)

                    spread_rate = base_spread_rate * \
                                  (1.0 + wind_effect + slope_effect + temp_effect_at_ij + humidity_effect_at_ij) * \
                                  fuel_at_ij

                    if spread_rate <= 0:
                        spread_prob = 0.0
                    else:
                        # Probability = 1 - exp(-rate * time)
                        spread_prob = 1.0 - np.exp(-spread_rate * time_step)

                    # Determine if fire spreads to this cell from this neighbor
                    if np.random.rand() < spread_prob: # Numba uses np.random.rand() for single float
                        new_fire[i, j] = 1
                        break # Cell (i,j) ignites, move to next cell (i,j+1) or (i+1,0)
    return new_fire


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
        print("Using Numba-optimized fire spread simulation...")
        # Extract relevant data maps
        wind_speed_map = weather_data['wind_speed']
        wind_direction_rad_map = np.radians(weather_data['wind_direction'])
        temperature_map = weather_data['temperature']
        humidity_map = weather_data['humidity']
        slope_deg_map = terrain_data['slope'] # Slope in degrees
        aspect_rad_map = np.radians(terrain_data['aspect']) # Aspect in radians

        # Simulation parameters (ensure these are floats for Numba)
        base_spread_rate = 0.3  # Base spread rate in km/h
        wind_factor = 0.5       # Wind influence factor
        slope_factor = 0.3      # Slope influence factor
        temp_factor = 0.2       # Temperature influence factor
        humidity_factor = -0.3  # Humidity influence factor (negative because higher humidity reduces spread)

        # Simulation loop
        for step in range(n_steps):
            if step == 0: # First call to Numba function can be slower due to JIT compilation
                print("Compiling Numba function for fire spread (first step may be slower)...")

            current_fire = _simulate_fire_spread_step_numba(
                current_fire, fuel_map,
                wind_speed_map, wind_direction_rad_map,
                temperature_map, humidity_map,
                slope_deg_map, aspect_rad_map,
                base_spread_rate, wind_factor, slope_factor,
                temp_factor, humidity_factor, float(time_step), # Ensure time_step is float
                NEIGHBORHOOD_OFFSETS, NEIGHBOR_ANGLES_RAD_CORRECTED # Use corrected angles
            )
            fire_states.append(current_fire.copy())
            if step == 0:
                 print("Numba compilation complete.")

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
# Main Workflow Functions
#######################

def setup_simulation_parameters(demo_mode=False):
    """Sets up dates, region, and synthetic data usage based on demo_mode."""
    print("Setting up simulation parameters...")
    if demo_mode:
        print("Running in demonstration mode with simplified parameters...")
        start_date = "2022-01-01"
        end_date = "2022-01-10"  # Shorter date range for demo
        effective_region = "Demo_" + REGION # For synthetic data grid size
        sim_hours = 6 # Shorter simulation for demo
    else:
        start_date = "2022-01-01"
        end_date = "2022-12-31"
        effective_region = REGION
        sim_hours = 12

    current_use_synthetic = USE_SYNTHETIC_DATA or demo_mode
    print(f"Data source: {'Synthetic' if current_use_synthetic else 'Actual (Not Implemented)'}")
    return start_date, end_date, effective_region, current_use_synthetic, sim_hours


def run_data_pipeline(effective_region_name, start_date_str, end_date_str, use_synthetic_data_flag):
    """Handles data collection and preprocessing."""
    print("\n--- Starting Data Pipeline ---")
    # 1. Data Collection
    weather_data = download_weather_data(effective_region_name if use_synthetic_data_flag else REGION, start_date_str, end_date_str, use_synthetic=use_synthetic_data_flag)
    terrain_data = download_terrain_data(effective_region_name if use_synthetic_data_flag else REGION, use_synthetic=use_synthetic_data_flag)
    lulc_data = download_lulc_data(effective_region_name if use_synthetic_data_flag else REGION, use_synthetic=use_synthetic_data_flag)
    human_factors = download_human_factors(effective_region_name if use_synthetic_data_flag else REGION, use_synthetic=use_synthetic_data_flag)
    historical_fires = download_historical_fires(effective_region_name if use_synthetic_data_flag else REGION, start_date_str, end_date_str, use_synthetic=use_synthetic_data_flag)

    if not use_synthetic_data_flag and (lulc_data is None or not weather_data or not terrain_data or not human_factors or not historical_fires):
        print("Error: Actual data download is not implemented or failed. Cannot proceed.")
        return None # Indicate failure

    # 2. Data Preprocessing
    # The fuel_map is created inside preprocess_data. We need it later for simulation.
    # Modify preprocess_data to return fuel_map if it's not already.
    # For now, let's assume preprocess_data structure is as is and we re-calculate fuel_map if needed,
    # or ideally, it should return it.
    # Let's adjust preprocess_data to return fuel_map.
    # For now, X, y = preprocess_data(...)
    # To get fuel_map, we'd need lulc_data.

    # Re-evaluating: preprocess_data already calculates fuel_map internally.
    # It is not directly returned but used to create feature `X`.
    # For the simulation, we need `fuel_map` separately.
    # It's better to calculate it once and pass it around.
    # For now, I will calculate it again in the main workflow if needed,
    # but a better refactor would be to have a dedicated fuel_map function or return it from preprocess.
    # Given the current plan, I will make a small adjustment to `preprocess_data` to return `fuel_map`.
    # This change is minor and related to this refactoring step.

    # Call the now primary preprocess_data function
    X, y, fuel_map_generated = preprocess_data(weather_data, terrain_data, lulc_data, human_factors, historical_fires)
    if X is None: # Preprocessing failed (preprocess_data returns None for X in this case)
        print("Error: Data preprocessing failed.")
        return None # Propagate failure

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)

    last_features_for_prediction = X[-1] if len(X) > 0 else None
    input_shape = X_train.shape[1:] if len(X_train) > 0 else (lulc_data.shape[0], lulc_data.shape[1], 13) # Fallback shape

    print("--- Data Pipeline Completed ---")
    return (weather_data, terrain_data, lulc_data, human_factors, historical_fires,
            X_train, y_train, X_val, y_val, X_test,
            last_features_for_prediction, input_shape, fuel_map_generated)


def train_or_load_prediction_model(input_shape, X_train, y_train, X_val, y_val, demo_mode_flag):
    """Creates and trains the fire prediction model, or uses a dummy for demo."""
    print("\n--- Starting Model Training/Setup ---")
    if demo_mode_flag:
        from tensorflow.keras.models import Sequential # Keep import local for demo
        from tensorflow.keras.layers import Conv2D
        print("Using simplified model for demonstration (not training)...")
        model = Sequential([
            Conv2D(16, 3, activation='relu', padding='same', input_shape=input_shape),
            Conv2D(8, 3, activation='relu', padding='same'),
            Conv2D(1, 1, activation='sigmoid', padding='same')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Skipping actual training for demonstration.")
    else:
        if X_train.size == 0 or y_train.size == 0:
            print("Error: Training data is empty. Cannot train model.")
            return None
        model = create_unet_model(input_shape)
        model, history = train_model(model, (X_train, y_train), (X_val, y_val), epochs=20) # epochs kept from original
    print("--- Model Training/Setup Completed ---")
    return model

def generate_fire_prediction(model, features_for_pred, demo_mode_flag, lulc_shape_for_demo):
    """Generates fire probability map using the model or dummy for demo."""
    print("\n--- Generating Fire Prediction ---")
    if demo_mode_flag:
        print("Generating dummy prediction for demonstration...")
        prediction_map = np.random.random(lulc_shape_for_demo)
    else:
        if model is None or features_for_pred is None:
            print("Error: Model or features for prediction not available. Cannot generate prediction.")
            # Fallback to a random map if in non-demo but something went wrong
            print("Falling back to random prediction map due to missing model/features.")
            prediction_map = np.random.random(lulc_shape_for_demo) # Needs a valid shape
        else:
            prediction_map = predict_fire_probability(model, features_for_pred)
    print("--- Fire Prediction Generated ---")
    return prediction_map


def run_fire_spread_simulation(initial_fire_map, weather_data_sim, terrain_data_sim, fuel_map_sim,
                               simulation_hours, sim_time_step, demo_mode_flag):
    """Runs the fire spread simulation."""
    print("\n--- Starting Fire Spread Simulation ---")
    fire_states_sim = simulate_fire_spread(
        initial_fire_map, weather_data_sim, terrain_data_sim, fuel_map_sim,
        hours=simulation_hours, time_step=sim_time_step, testing_mode=demo_mode_flag
    )
    print("--- Fire Spread Simulation Completed ---")
    return fire_states_sim

def visualize_and_save_results(pred_map, spread_states, sim_time_step_vis, demo_mode_flag, output_directory):
    """Visualizes and saves all results (images, GeoTIFFs)."""
    print("\n--- Visualizing and Saving Results ---")
    # Visualize prediction
    visualize_prediction(pred_map, os.path.join(output_directory, 'fire_prediction.png'), testing_mode=demo_mode_flag)
    save_as_geotiff(pred_map, os.path.join(output_directory, 'fire_prediction.tif'))

    # Create animation for spread
    create_fire_spread_animation(
        spread_states, sim_time_step_vis,
        os.path.join(output_directory, 'fire_spread_animation.gif'),
        testing_mode=demo_mode_flag
    )
    # Save fire spread states as GeoTIFFs
    for hour_save in [1, 2, 3, 6, 12]: # Standard hours to save
        # Ensure the hour is within the simulated states and less than or equal to max sim hours
        if hour_save < len(spread_states) and hour_save <= (len(spread_states) -1) * sim_time_step_vis :
             save_as_geotiff(
                spread_states[hour_save // sim_time_step_vis], # Index is step, step = hour / time_step
                os.path.join(output_directory, f'fire_spread_{hour_save}h.tif')
            )
    print("--- Visualization and Saving Completed ---")


# Original preprocess_data is no longer called directly;
# preprocess_data_extended has become the main version.
# Renaming preprocess_data_extended to preprocess_data for clarity.

def preprocess_data(weather_data, terrain_data, lulc_data, human_factors, historical_fires):
    """
    Preprocess and combine all data sources into feature stacks for model training.
    Also returns the generated fuel_map.

    Args:
        weather_data (dict): Weather data rasters
        terrain_data (dict): Terrain data rasters
        lulc_data (np.ndarray): LULC classification raster
        human_factors (dict): Human factors rasters
        historical_fires (dict): Historical fire data

    Returns:
        tuple: (X, y, fuel_map)
               X (np.ndarray): Feature array for model training
               y (np.ndarray): Target array for model training
               fuel_map (np.ndarray): Fuel availability map
               Returns (None, None, None) or (empty_arrays, ..., fuel_map) on failure/empty data.
    """
    print("Preprocessing data...")
    fuel_map = np.zeros_like(lulc_data, dtype=float)
    fuel_values = {
        0: 0.0, 1: 0.2, 2: 0.4, 3: 0.7, 4: 0.8,
        5: 0.9, 6: 1.0, 7: 0.95, 8: 0.1
    }
    for lulc_class, fuel_value in fuel_values.items():
        fuel_map[lulc_data == lulc_class] = fuel_value

    if not weather_data or not terrain_data or not human_factors: # LULC checked by caller of run_data_pipeline
        print("Error: Weather, terrain, or human factors data is missing in preprocess_extended.")
        # Fallback: Try to determine shape from lulc_data if possible for empty array dimensions
        h, w = lulc_data.shape if lulc_data is not None else (10,10) # Default small shape
        num_feat = 13 # Default number of features
        return np.empty((0,h,w,num_feat)), np.empty((0,h,w)), fuel_map if lulc_data is not None else np.zeros((h,w))


    wind_dir_rad = np.radians(weather_data['wind_direction'])
    aspect_rad = np.radians(terrain_data['aspect'])
    sin_wind_direction = np.sin(wind_dir_rad)
    cos_wind_direction = np.cos(wind_dir_rad)
    sin_aspect = np.sin(aspect_rad)
    cos_aspect = np.cos(aspect_rad)

    static_feature_layers = [
        weather_data['temperature'], weather_data['precipitation'], weather_data['humidity'],
        weather_data['wind_speed'], sin_wind_direction, cos_wind_direction,
        terrain_data['slope'], sin_aspect, cos_aspect,
        fuel_map, # fuel_map is now part of static features
        human_factors['settlement'], human_factors['roads'],
    ]
    static_features_stack = np.stack(static_feature_layers, axis=-1)

    X_list, y_list = [], []
    if not historical_fires:
        print("Warning: Historical fire data is empty in preprocess_extended.")
        dates = []
    else:
        dates = sorted(historical_fires.keys())

    if not dates or len(dates) < 2:
        print("Warning: Not enough historical fire data for training samples in preprocess_extended.")
        num_features = static_features_stack.shape[-1] + 1
        height, width = fuel_map.shape
        return np.empty((0, height, width, num_features)), np.empty((0, height, width)), fuel_map

    for i in range(len(dates) - 1):
        current_date, next_date = dates[i], dates[i+1]
        current_fire_state = historical_fires[current_date].astype(float)
        current_fire_state_reshaped = current_fire_state[..., np.newaxis]
        features_for_date = np.concatenate((static_features_stack, current_fire_state_reshaped), axis=-1)
        X_list.append(features_for_date)
        y_list.append(historical_fires[next_date].astype(float))

    if not X_list:
        num_features = static_features_stack.shape[-1] + 1
        height, width = fuel_map.shape
        return np.empty((0, height, width, num_features)), np.empty((0, height, width)), fuel_map

    return np.array(X_list), np.array(y_list), fuel_map


#######################
# Main Function
#######################

def main(demo_mode=False):
    """
    Main function to run the forest fire prediction and simulation.
    """
    print("Starting Forest Fire Prediction and Simulation...")

    start_date, end_date, effective_region, current_use_synthetic, sim_hours = setup_simulation_parameters(demo_mode)

    data_pipeline_results = run_data_pipeline(effective_region, start_date, end_date, current_use_synthetic)
    if data_pipeline_results is None:
        print("Exiting due to data pipeline failure.")
        return

    (weather_data, terrain_data, lulc_data, _, _, # human_factors, historical_fires are not directly needed after pipeline
     X_train, y_train, X_val, y_val, _, # X_test not used further in this main flow
     last_features_for_prediction, model_input_shape, fuel_map) = data_pipeline_results

    # Ensure lulc_data is available for fallback shape if prediction fails
    lulc_shape_for_demo = lulc_data.shape if lulc_data is not None else (sdg.get_grid_size(effective_region, RESOLUTION), sdg.get_grid_size(effective_region, RESOLUTION))


    model = train_or_load_prediction_model(model_input_shape, X_train, y_train, X_val, y_val, demo_mode)

    # Handle case where model training might fail or be skipped (e.g. no data)
    if model is None and not demo_mode: # if demo_mode, model is dummy but valid
        print("Model could not be trained/loaded. Proceeding with dummy prediction for simulation.")
        # Create a dummy prediction if model is None and not in demo_mode (where it's handled)
        # This ensures simulation can run with some input.
        prediction_map = np.random.random(lulc_shape_for_demo)
    else:
        prediction_map = generate_fire_prediction(model, last_features_for_prediction, demo_mode, lulc_shape_for_demo)


    # Fire Spread Simulation
    initial_fire = (prediction_map > 0.8).astype(int) # Use threshold from original main
    sim_time_step = 1  # 1 hour, from original main

    fire_states = run_fire_spread_simulation(
        initial_fire, weather_data, terrain_data, fuel_map, # Use the fuel_map from data_pipeline
        sim_hours, sim_time_step, demo_mode
    )

    visualize_and_save_results(prediction_map, fire_states, sim_time_step, demo_mode, OUTPUT_DIR)

    print("\nForest Fire Prediction and Simulation completed successfully!")

if __name__ == "__main__":
    # Run in demo mode for faster execution and lower memory usage
    main(demo_mode=True)
