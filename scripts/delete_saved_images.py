import os

SAVED_IMAGES_PATH = 'app/static/uploaded_images/'

if __name__ == '__main__':

    for image in os.scandir(SAVED_IMAGES_PATH):
        os.remove(image)
    print("Cleared Saved Images.")

    

    