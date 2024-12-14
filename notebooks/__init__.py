"""
ruf notebooks module
"""

from dotenv import load_dotenv
import os

load_dotenv()

# Check if the environment variable is loaded
print(os.getenv("PLANTSENSE_IMAGES_DATABASE_URI"))