import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from datetime import datetime, timedelta

RESOLUTION = 30 # Default resolution, can be overridden

def generate_synthetic_weather_data(grid_size, start_date_str, end_date_str):
    """Generates synthetic weather data."""
    temperature = np.random.normal(30, 5, (grid_size, grid_size))
    precipitation = np.random.exponential(2, (grid_size, grid_size))
    humidity = np.random.normal(60, 15, (grid_size, grid_size)).clip(10, 100)
    wind_speed = np.random.rayleigh(4, (grid_size, grid_size))
    wind_direction = np.random.uniform(0, 360, (grid_size, grid_size))
    return {
        'temperature': temperature,
        'precipitation': precipitation,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction
    }

def generate_synthetic_terrain_data(grid_size):
    """Generates synthetic terrain data (DEM, slope, aspect)."""
    x = np.linspace(0, 10, grid_size)
    y = np.linspace(0, 10, grid_size)
    X, Y = np.meshgrid(x, y)
    peaks = [
        (3, 3, 2000, 1), (7, 7, 3000, 1.5),
        (2, 8, 2500, 0.8), (8, 2, 1800, 1.2)
    ]
    dem = np.zeros((grid_size, grid_size))
    for px, py, height, spread in peaks:
        dem += height * np.exp(-((X - px)**2 + (Y - py)**2) / spread)

    dy, dx = np.gradient(dem, RESOLUTION, RESOLUTION)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    aspect = np.degrees(np.arctan2(dy, -dx))
    aspect = np.where(aspect < 0, aspect + 360, aspect)
    return {'dem': dem, 'slope': slope, 'aspect': aspect}

def generate_synthetic_lulc_data(grid_size):
    """Generates synthetic LULC data."""
    prob_maps = [gaussian_filter(np.random.random((grid_size, grid_size)), sigma=5) for _ in range(9)]
    lulc = np.argmax(np.array(prob_maps), axis=0)
    return lulc

def generate_synthetic_human_factors(grid_size):
    """Generates synthetic human factors data (settlements, roads)."""
    settlement = np.zeros((grid_size, grid_size))
    centers = [
        (int(grid_size * 0.2), int(grid_size * 0.3), 0.8),
        (int(grid_size * 0.7), int(grid_size * 0.2), 0.6),
        (int(grid_size * 0.5), int(grid_size * 0.5), 1.0),
        (int(grid_size * 0.8), int(grid_size * 0.8), 0.7)
    ]
    for cx, cy, intensity in centers:
        y_coords, x_coords = np.ogrid[-cy:grid_size-cy, -cx:grid_size-cx]
        mask = x_coords*x_coords + y_coords*y_coords <= (grid_size/10)**2
        dist_squared = x_coords*x_coords + y_coords*y_coords
        settlement[mask] = np.maximum(settlement[mask], intensity * np.exp(-(dist_squared[mask] / (grid_size/15)**2)))

    roads = np.zeros((grid_size, grid_size))
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            x1, y1, _ = centers[i]
            x2, y2, _ = centers[j]
            t = np.linspace(0, 1, 100)
            noise = np.random.normal(0, grid_size/20, (100, 2))
            road_x = (x1 * (1-t) + x2 * t + noise[:, 0]).astype(int)
            road_y = (y1 * (1-t) + y2 * t + noise[:, 1]).astype(int)
            mask = (road_x >= 0) & (road_x < grid_size) & (road_y >= 0) & (road_y < grid_size)
            roads[road_y[mask], road_x[mask]] = 1
    roads = binary_dilation(roads, iterations=2)
    return {'settlement': settlement, 'roads': roads}

def generate_synthetic_historical_fires(grid_size, start_date_str, end_date_str):
    """Generates synthetic historical fire data."""
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    date_range = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    fire_data = {}
    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        month = date.month
        fire_probability = 0.001 if 4 <= month <= 6 else 0.0002
        fire_map = np.random.random((grid_size, grid_size)) < fire_probability
        fire_map = binary_dilation(fire_map, iterations=2)
        fire_data[date_str] = fire_map
    return fire_data

def get_grid_size(region_name_for_synthetic_data, default_resolution):
    """
    Determines grid size based on region name.
    "Test" in region name uses a smaller grid.
    """
    global RESOLUTION
    RESOLUTION = default_resolution
    if "Test" in region_name_for_synthetic_data:
        return int(10000 / RESOLUTION)  # Smaller grid for testing (10x10 km)
    else:
        return int(100000 / RESOLUTION) # Full size for actual runs (100x100 km)
