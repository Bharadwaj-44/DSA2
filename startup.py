# Startup configuration for DSA kernel
import os
import time
import hashlib

# Set matplotlib backend before any other imports
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure matplotlib
plt.ioff()  # Turn off interactive mode
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['savefig.bbox'] = 'tight'

# Custom show function
def custom_show(*args, **kwargs):
    """Custom show function that saves plots automatically"""
    fig = plt.gcf()
    if fig.get_axes():  # Only save if there are axes
        # Generate a unique filename
        timestamp = str(time.time())
        filename = f"{hashlib.md5(timestamp.encode()).hexdigest()}.png"
        
        # Get the session cache path from environment
        cache_path = os.environ.get('DSA_SESSION_CACHE_PATH', './cache')
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        
        filepath = os.path.join(cache_path, filename)
        try:
            fig.savefig(filepath, dpi=100, bbox_inches='tight')
            print(f"Plot saved: {filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        print("No axes found in figure, skipping save")

# Replace plt.show with our custom function
original_show = plt.show
plt.show = custom_show

# Set matplotlib to not show warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Configure pandas display options
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Import seaborn to ensure it's available
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    print("Seaborn imported successfully")
except ImportError:
    print("Warning: Seaborn not available")

# Import scikit-learn for ML models
try:
    import sklearn
    print("Scikit-learn imported successfully")
except ImportError:
    print("Warning: Scikit-learn not available")

# Pre-import heavy packages to avoid runtime delays
try:
    import statsmodels
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    print("Statsmodels imported successfully")
except ImportError:
    print("Warning: Statsmodels not available")

try:
    import scipy
    import scipy.stats
    print("Scipy imported successfully")
except ImportError:
    print("Warning: Scipy not available")

try:
    import numpy as np
    print("Numpy imported successfully")
except ImportError:
    print("Warning: Numpy not available")

try:
    import joblib
    print("Joblib imported successfully")
except ImportError:
    print("Warning: Joblib not available")

print("DSA kernel initialized successfully!")
