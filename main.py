import glob
import os
import sys

# Ensure custom modules (model, agent) are discoverable on local and Kaggle environments
if "__file__" in globals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = os.getcwd()

if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add nested src/ folder to path for local execution compatibility
src_dir = os.path.join(current_dir, "src")
if os.path.exists(src_dir) and src_dir not in sys.path:
    sys.path.append(src_dir)

kaggle_path = '/kaggle_simulations/agent'
if os.path.exists(kaggle_path) and kaggle_path not in sys.path:
    sys.path.append(kaggle_path)

# Dynamically resolve and append cg-lib
try:
    cg_lib_path = glob.glob('/kaggle/input/**/cg-lib', recursive=True)[0]
    sys.path.append(cg_lib_path)
except IndexError:
    pass

# Expose agent function for the competition execution environment
from agent import agent
