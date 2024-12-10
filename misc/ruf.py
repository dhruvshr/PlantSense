import os
import dotenv

dotenv.load_dotenv()

DATA_DRIVE = os.getenv('DATA_DRIVE')

def rename_images_in_directory(directory_path):
    # Get all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file has a space in its name
        if ' ' in filename:
            # Create the new filename by removing spaces
            new_filename = filename.replace(' ', '_')  # Replace space with underscore, or you can remove it entirely with filename.replace(' ', '')
            
            # Construct full file paths
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

# Example usage

plant_village_path = DATA_DRIVE + 'Plant_leave_diseases_dataset_with_augmentation/'

for directory in os.listdir(plant_village_path):
    directory_path = os.path.join(plant_village_path, directory)
    rename_images_in_directory(directory_path)
