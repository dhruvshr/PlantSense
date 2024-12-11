"""
Config Utils for ML src
"""

import os


def set_env_variable(key, value, env_file=".env"):
    # check if the .env file exists
    if os.path.exists(env_file):
        with open(env_file, 'r') as file:
            lines = file.readlines()
        
        # check if the key already exists in the file
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines = f"{key}={value}\n"
                break
        else:
            # if key doesn't exist, add it at the end
            lines.append(f"{key}={value}\n")
        
        # write the updated content back to the .env file
        with open(env_file, 'w') as file:
            file.writelines(lines)
    else:
        # If the .env file doesn't exist, create a new one
        with open(env_file, 'w') as file:
            file.write(f"{key}={value}\n")