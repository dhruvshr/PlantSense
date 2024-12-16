"""
ML Model Inference script to handle image inference and conversational inference
"""

import os
from src.datasets.plant_village import PlantVillage
from src.llm.insights_engine import InsightsEngine
from src.utils.device import get_device
from src.utils.inference import infer_image
from src.utils.model_loader import load_model

def main():
    # Device configuration
    device = get_device()

    base_dataset = PlantVillage()
    # Load the trained model
    model = load_model(device)

    insights_engine = InsightsEngine()

    print("Welcome to the PlantSense CLI!")
    print("Please upload an image of your plant, and I'll analyze it and provide insights.")

    while True:
        user_input = input("Enter 'upload' to upload an image, or 'exit' to quit: ")

        if user_input.lower() == 'upload':
            image_path = input("Enter the path to the image file: ")
            if os.path.exists(image_path):
                # perform inference
                predicted_class, confidence_score = infer_image(image_path, model, device)
                # Create prediction dictionary with both class and confidence
                # prediction = [{'class': predicted_class, 'confidence': confidence_score}]  # Adding mock confidence for now
                insights = insights_engine.generate_insights(predicted_class, confidence_score)  # Wrap in lists
                # print(f"The plant disease detected is: {predicted_class}")
                print(f"PlantSense: {insights}")

                while True:
                        # user_feedback = input("Do you have any questions about the results? (Enter 'yes' or 'no'): ")
                        user_feedback = input("\nUser: ")

                        if user_feedback:
                            if user_feedback == 'no':
                                print("\nThank you! If you have another plant to analyze, feel free to upload another image.")
                                break

                            # user_question = input("Please enter your question: ")
                            # generate conversational insights for user feedback
                            follow_up_insights = insights_engine.generate_insights(predicted_class, confidence_score, user_feedback)
                            print(f"PlantSense: {follow_up_insights}")
                            
                        else:
                            print("Invalid input. Please enter 'yes' or 'no'.")
            else:
                print("The image file could not be found. Please check the path and try again.")
        elif user_input.lower() == 'exit':
            print("\nExiting the plant disease detection CLI. Goodbye!")
            break
        else:
            print("Invalid input. Please try again.")

if __name__ == '__main__':
    main()