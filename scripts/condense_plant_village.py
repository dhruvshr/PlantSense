import os
import random
import shutil

def delete_percentage_of_images(dataset_path, target_class, percentage_to_delete=0.6):
    """
    Delete a specified percentage of images from a specific class in the dataset.
    
    Args:
        dataset_path (str): Path to the Plant Village dataset
        target_class (str): Name of the class to delete images from
        percentage_to_delete (float): Percentage of images to delete (0.0 to 1.0)
    """
    class_path = os.path.join(dataset_path, target_class)
    
    if not os.path.exists(class_path):
        print(f"Error: Class '{target_class}' not found in the dataset.")
        return
    
    # Get all image files in the class directory
    image_files = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Calculate number of images to delete
    num_images = len(image_files)
    num_to_delete = int(num_images * percentage_to_delete)
    
    # Randomly select images to delete
    images_to_delete = random.sample(image_files, num_to_delete)
    
    print(f"Class: {target_class}")
    print(f"Total images: {num_images}")
    print(f"Deleting: {num_to_delete} images")
    
    # Delete selected images
    for image in images_to_delete:
        image_path = os.path.join(class_path, image)
        os.remove(image_path)
        
    print(f"Remaining images: {len(os.listdir(class_path))}\n")

def main():

    dataset_path = "data/raw/Plant_leave_diseases_dataset_with_augmentation/"
    
    class_dirs = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
    
    print("Available classes:")
    for i, class_name in enumerate(class_dirs):
        print(f"{i+1}. {class_name}")
    
    class_index = int(input("\nEnter the number of the class to process: ")) - 1
    target_class = class_dirs[class_index]
    
    print(f"\nThis script will permanently delete 60% of images from the class: {target_class}")
    confirmation = input("Do you want to proceed? (yes/no): ")
    
    if confirmation.lower() == 'yes':
        delete_percentage_of_images(dataset_path, target_class)
        print("Deletion complete!")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()
