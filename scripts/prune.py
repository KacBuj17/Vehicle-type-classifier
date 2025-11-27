import torch.nn.utils.prune as prune
import torch
import json
import os

from data_loader import get_data_loaders
from eval_utils import test_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = torch.load("resources/models/base_model.pth", map_location=device, weights_only=False)
    model.to(device)

    parameters_to_prune = [
        (model.conv1, "weight"),
        (model.conv2, "weight"),
        (model.conv3, "weight"),
        (model.fc1, "weight"),
        (model.fc2, "weight"),
        (model.fc3, "weight"),
    ]

    print("Before pruning:")
    for module, _ in parameters_to_prune:
        weight = module.weight.data
        n_params = weight.numel()
        n_zeros = torch.sum(weight == 0).item()
        print(f"{module._get_name()}: {n_zeros}/{n_params} zeros ({100 * n_zeros / n_params:.2f}%)")

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.5,
    )

    print("\nAfter pruning:")
    for module, _ in parameters_to_prune:
        module.weight.data *= module.weight_mask
        n_params = module.weight.numel()
        n_zeros = torch.sum(module.weight.data == 0).item()
        print(f"{module._get_name()}: {n_zeros}/{n_params} zeros ({100 * n_zeros / n_params:.2f}%)")
        prune.remove(module, "weight")

    _, test_loader = get_data_loaders(
        dataset_path="resources/data/VehiclesDataset",
        batch_size=64
    )

    accuracy, precision, recall, f1 = test_model(model, test_loader, device)
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    stats_filepath = f"resources/models/pruned_model_stats.json"

    stats = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    os.makedirs(os.path.dirname(stats_filepath), exist_ok=True) if os.path.dirname(stats_filepath) else None

    with open(stats_filepath, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    torch.save(model, "resources/models/pruned_model.pth")


if __name__ == "__main__":
    main()
