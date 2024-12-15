"""
ML Model Evaluation script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from src.utils.device import get_device
from src.datasets.plant_village import PlantVillage
from src.evaluation.evaluator import ModelEvaluator
from src.model.atomic_resnet import ResNet
from src.utils.metrics import MetricsTracker

# define model path
MODEL_PATH = "saved_models/checkpoints/resnet_model_best_2.pth"

matplotlib.use('Agg')


def main():
    # Device configuration
    device = get_device()    
    
    # Load test dataset
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    base_dataset = PlantVillage(transform=test_transform)

    # for i in range(3):
    #     print(f"Dataset {i+1}: {len(base_dataset.split_dataset()[i])}")

    test_dataset = base_dataset.split_dataset()[2]

    # test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    model = ResNet(num_classes=PlantVillage().NUM_CLASSES).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    model.eval()

    # initialize evaluator
    evaluator = ModelEvaluator(
        model=model, 
        device=device, 
        test_loader=test_loader,
        class_names=base_dataset.plantvillage.classes
    )

    print(f"Evaluating {model.__class__.__name__} on {device}")
    
    # perform evaluation
    print("Model Performance Metrics:")
    metrics = evaluator.compute_metrics()
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.2f}")
    
    # top-k accuracy
    top_3_accuracy = evaluator.top_k_accuracy(k=3)
    print(f"\nTop-3 Accuracy: {top_3_accuracy:.2f}%")
    
    # visualize confusion matrix
    cm_fig = evaluator.plot_confusion_matrix(normalize=True)
    cm_fig.savefig('results/confusion_matrix.png')
    plt.close(cm_fig)

    # update metrics
    # evaluator.metrics.plot_metrics(model=model, save_path='results/metrics.png')

if __name__ == '__main__':
    main()


