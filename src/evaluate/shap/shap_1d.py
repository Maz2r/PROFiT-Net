import os
import sys

# Add project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import numpy as np
import torch
import pandas as pd
import shap
import matplotlib.pyplot as plt
from src.models.pytorch.cnn_model import CNNModel
from src.utils.target_labels import TargetLabels
import wandb

def get_shap_values_1d(target_abbreviation, run_id):
    # Initialize WandB
    wandb.init(
        project="SePROFiT-Net",  # Replace with your WandB project name
        name=f"shap_1d_{target_abbreviation}_{run_id}",
        entity='cnmd-phb-postech'
    )

    # Map abbreviation to full target name
    targets = TargetLabels.get_all_targets()
    if target_abbreviation not in targets:
        print(f"Error: Invalid target abbreviation '{target_abbreviation}'.")
        wandb.finish()
        return

    target_full_name = targets[target_abbreviation]
    data_dir = os.path.join(os.getcwd(), 'data', target_full_name)

    # Validate data directory
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist.")
        wandb.finish()
        return

    # Load test data
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    # Load column names from CSV file and ignore 'id' and 'target' columns
    csv_path = os.path.join(os.getcwd(), 'dataset', 'exp_band_gap_2', f"exp_band_gap.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        feature_names = [col for col in df.columns if col not in ['id', 'target']]
    else:
        feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]  # Default to generic names

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fetch model weights from WandB
    artifact_name = f"{target_abbreviation}_{run_id}:latest"
    artifact = wandb.use_artifact(artifact_name, type='model')
    artifact_dir = artifact.download()

    # Load model
    checkpoint_path = os.path.join(artifact_dir, f"{target_abbreviation}_{run_id}_cp.pt")
    model = CNNModel(X_test.shape[1]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    # SHAP Analysis
    explainer = shap.DeepExplainer(model, torch.tensor(X_test[:100]).float().unsqueeze(1).to(device))  # Use a subset
    shap_values = explainer.shap_values(torch.tensor(X_test).float().unsqueeze(1).to(device))

    # Create directory for saving SHAP results
    save_dir = os.path.abspath(os.path.join("images/SHAP", target_abbreviation))
    os.makedirs(save_dir, exist_ok=True)

    # Save SHAP values to CSV
    shap_values_flat = np.squeeze(shap_values)  # Remove unnecessary dimensions
    shap_df = pd.DataFrame(shap_values_flat, columns=feature_names)
    csv_path = os.path.join(save_dir, f"shap_values_{target_abbreviation}.csv")
    shap_df.to_csv(csv_path, index=False)
    print(f"SHAP values saved to {csv_path}")

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values_flat), axis=0)
    top_10_indices = np.argsort(mean_abs_shap)[-10:][::-1]
    top_10_features = [feature_names[i] for i in top_10_indices]
    top_10_values = mean_abs_shap[top_10_indices]

    # Create bar graph for top 10 features
    plt.figure(figsize=(10, 6))
    plt.barh(top_10_features, top_10_values, color='skyblue')
    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Feature')
    plt.title('Top 10 Features by SHAP Value')
    plt.gca().invert_yaxis()  # Highest value at the top

    # Annotate each bar with its corresponding SHAP value
    for i, (feature, value) in enumerate(zip(top_10_features, top_10_values)):
        plt.text(value, i, f"{value:.4f}", va='center')

    # Save the bar graph
    bar_graph_path = os.path.join(save_dir, f"shap_top10_{target_abbreviation}.png")
    plt.savefig(bar_graph_path, bbox_inches='tight')
    print(f"Top 10 features bar graph saved to {bar_graph_path}")

    # Generate heatmap of SHAP values divided into 8x8 squares
    reshaped_shap_values = shap_values_flat.reshape(-1, 136, 136)
    mean_shap_values = np.mean(reshaped_shap_values, axis=0)
    mean_shap_8x8 = mean_shap_values.reshape(8, 17, 8, 17).mean(axis=(1, 3))
    plt.figure(figsize=(8, 8))
    plt.imshow(mean_shap_8x8, cmap="viridis", extent=[0, 136, 136, 0])
    plt.colorbar(label='Mean SHAP Value')
    plt.title("SHAP Heatmap (8x8 Grid)")
    heatmap_path = os.path.join(save_dir, f"shap_heatmap_{target_abbreviation}.png")
    plt.savefig(heatmap_path, bbox_inches='tight')
    print(f"SHAP heatmap saved to {heatmap_path}")

    wandb.finish()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python shap_1d.py <target_abbreviation> <run_id>")
        sys.exit(1)

    target_abbreviation = sys.argv[1]
    run_id = sys.argv[2]

    get_shap_values_1d(target_abbreviation, run_id)
