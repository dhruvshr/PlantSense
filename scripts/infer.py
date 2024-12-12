"""
ML Model Inference script to handle image inference and conversational inference
"""

import os
import torch
from torchvision import transforms
from PIL import Image
from src.datasets.plant_village import PlantVillage
from src.model.plantsense_resnet import PlantSenseResNetBase
from src.evaluation.evaluator import ModelEvaluator

# def
BATCH_SIZE = 32
NUM_WORKERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
MODEL_V1_1_PATH = 'model_files/modelv1_1.pth'

def load_model(device, base_dataset):
    # load model (modify as per training script)
    model = PlantSenseResNetBase(
        num_classes=len(base_dataset.plantvillage.classes),
    ).to(device)

    # Load the state dict and modify the keys to match your model structure
    state_dict = torch.load(MODEL_V1_1_PATH, weights_only=True)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[f"base_model.{key}"] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    return model

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dataset = PlantVillage()
    # Load the trained model
    model = load_model(device, base_dataset)

    # Initialize the evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        test_loader=None,
        class_names=base_dataset.plantvillage.classes
    )

    print("Welcome to the Plant Disease Detection CLI!")
    print("Please upload an image of your plant, and I'll analyze it and provide insights.")

    while True:
        user_input = input("Enter 'upload' to upload an image, or 'exit' to quit: ")

        if user_input.lower() == 'upload':
            image_path = input("Enter the path to the image file: ")
            if os.path.exists(image_path):
                image = Image.open(image_path)
                image_tensor = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])(image).unsqueeze(0).to(device)

                predicted_class, insights = evaluator.evaluate_and_explain(image_tensor)
                print(f"The plant disease detected is: {predicted_class}")
                print(f"Actionable insights: {insights}")

                user_feedback = input("Do you have any questions or comments about the results? (Enter 'yes' or 'no'): ")
                if user_feedback.lower() == 'yes':
                    user_question = input("Please enter your question or comment: ")
                    # TODO: Implement conversational logic to handle user feedback
                    print("Thank you for your feedback. I'll do my best to provide more information.")
            else:
                print("The image file could not be found. Please check the path and try again.")
        elif user_input.lower() == 'exit':
            print("Exiting the plant disease detection CLI. Goodbye!")
            break
        else:
            print("Invalid input. Please try again.")

if __name__ == '__main__':
    main()