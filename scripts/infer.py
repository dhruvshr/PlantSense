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
from src.llm.insights_engine import InsightsEngine
from src.utils.device import get_device

# def
BATCH_SIZE = 32
NUM_WORKERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
MODEL_V1_1_PATH = 'saved_models/modelv1_1.pth'

def infer_image(image_path, model, device, base_dataset):
    # load image
    image = Image.open(image_path)
    image_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])(image).unsqueeze(0).to(device)

    # perform inference
    with torch.no_grad():
        output = model(image_tensor)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = base_dataset.plantvillage.classes[predicted_idx]
        # Convert confidence to percentage
        confidence_score = confidence.item() * 100

    return predicted_class, confidence_score

def load_model(device):
    # load model (modify as per training script)
    model = PlantSenseResNetBase(
        num_classes=PlantVillage().NUM_CLASSES,
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
    device = get_device()

    base_dataset = PlantVillage()
    # Load the trained model
    model = load_model(device)

    # Initialize the evaluator
    # evaluator = ModelEvaluator(
    #     model=model,
    #     device=device,
    #     test_loader=None,
    #     class_names=base_dataset.plantvillage.classes
    # )

    insights_engine = InsightsEngine()

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

                predicted_class, confidence_score = infer_image(image_path, model, device, base_dataset)
                # Create prediction dictionary with both class and confidence
                # prediction = [{'class': predicted_class, 'confidence': confidence_score}]  # Adding mock confidence for now
                insights = insights_engine.generate_insights(predicted_class, confidence_score)  # Wrap in lists
                print(f"The plant disease detected is: {predicted_class}")
                print(f"PlantSense: {insights}")

                user_feedback = input("Do you have any questions or comments about the results? (Enter 'yes' or 'no'): ")
                if user_feedback.lower() == 'yes':
                    user_question = input("Please enter your question or comment: ")
                    # TODO: implement conversational logic to handle user feedback
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