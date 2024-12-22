import os
import random
import shutil
import sys
import subprocess
import ctypes

# ONLY FOR CODE DELIVERY AND SUBMISSION TO CANVAS

def check_sudo_permission():
    """Check if the script has sudo privileges"""
    if os.name == 'nt':  # Windows
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    else:  # Unix-like
        return os.geteuid() == 0

def restart_with_sudo():
    """Restart the script with sudo privileges"""
    if os.name == 'nt':  # Windows
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    else:  # Unix-like
        args = ['sudo'] + [sys.executable] + sys.argv
        subprocess.run(args)

def balance_class_images(dataset_path, target_class, target_count=150, tolerance=5):
    """
    Balance a class to have approximately target_count images (±tolerance).
    
    Args:
        dataset_path (str): Path to the Plant Village dataset
        target_class (str): Name of the class to balance
        target_count (int): Desired number of images
        tolerance (int): Acceptable deviation from target count
    """
    class_path = os.path.join(dataset_path, target_class)
    
    if not os.path.exists(class_path):
        print(f"Error: Class '{target_class}' not found in the dataset.")
        return
    
    # Get all image files in the class directory
    image_files = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    current_count = len(image_files)
    
    if current_count < target_count - tolerance:
        print(f"Warning: {target_class} has only {current_count} images, which is below target range.")
        return
    
    if current_count > target_count + tolerance:
        # Calculate number of images to delete
        num_to_delete = current_count - target_count
        images_to_delete = random.sample(image_files, num_to_delete)
        
        print(f"\nClass: {target_class}")
        print(f"Current images: {current_count}")
        print(f"Deleting: {num_to_delete} images")
        
        # Delete selected images
        for image in images_to_delete:
            image_path = os.path.join(class_path, image)
            os.remove(image_path)
        
        print(f"Remaining images: {len(os.listdir(class_path))}")

def main():
    # Check for sudo privileges
    if not check_sudo_permission():
        print("This script requires administrative privileges to delete files.")
        print("Restarting with elevated privileges...")
        restart_with_sudo()
        sys.exit(0)

    dataset_path = "data/raw/Plant_leave_diseases_dataset_with_augmentation/"
    
    class_dirs = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
    
    print("Available classes:")
    for i, class_name in enumerate(class_dirs):
        print(f"{i+1}. {class_name}")
    
    print("\nThis script will balance all classes to have 150 (±5) images.")
    confirmation = input("Do you want to proceed? (yes/no): ")
    
    if confirmation.lower() == 'yes':
        for class_name in class_dirs:
            balance_class_images(dataset_path, class_name)
        print("\nBalancing complete!")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()
