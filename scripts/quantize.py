import torch
import json
import os

from eval_utils import test_model
from data_loader import get_data_loaders


def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    model_path = "resources/models/pruned_model.pth"
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    print("\nWeights before quantization:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(name, module.weight.dtype)

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    print("\nWeights after quantization:")
    for name, module in quantized_model.named_modules():
        if isinstance(module, torch.nn.quantized.dynamic.Linear):
            dtype = module._packed_params._weight_bias()[0].dtype
            print(f"{name}: {dtype}")

    _, test_loader = get_data_loaders(
        dataset_path="resources/data/VehiclesDataset",
        batch_size=64
    )

    accuracy, precision, recall, f1 = test_model(model, test_loader, device)
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    stats_filepath = f"resources/models/quantized_model_stats.json"

    stats = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    os.makedirs(os.path.dirname(stats_filepath), exist_ok=True) if os.path.dirname(stats_filepath) else None

    with open(stats_filepath, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    torch.save(quantized_model, "resources/models/pruned_quantized_model.pth")


if __name__ == "__main__":
    main()
