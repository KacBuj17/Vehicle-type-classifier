import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

from eval_utils import test_model
from model import SimpleCNN
from data_loader import get_data_loaders
from model_trainer import ModelTrainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(torch.__version__)
    print(f"Device used: {device}")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader, test_loader = get_data_loaders(
        dataset_path="resources/data/VehiclesDataset", batch_size=64
    )

    model.to(device)
    trainer = ModelTrainer(model, train_loader, test_loader, criterion, optimizer, device)
    trainer.train(10)

    accuracy, precision, recall, f1 = test_model(model, test_loader, device)
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    stats_filepath = f"resources/models/base_model_stats.json"

    stats = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    os.makedirs(os.path.dirname(stats_filepath), exist_ok=True) if os.path.dirname(stats_filepath) else None

    with open(stats_filepath, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    torch.save(model, "resources/models/base_model.pth")


if __name__ == "__main__":
    main()
