import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(history, save=False):

    plt.figure(figsize=(10,6))

    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)


    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)",fontsize=12)
    plt.title("Learning Curve", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)


    if save:
        plt.savefig("Learning_curve.png", dpi=300, bbox_inches="tight")

    plt.show()

def plot_predictions_vs_reality(y_true, y_pred, save=False):
    
    
    plt.figure(figsize=(10,10))

    plt.scatter(y_true, y_pred, alpha=0.5, s=30)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")


    plt.xlabel("Real Consumption (kWh/dia)", fontsize=12)
    plt.ylabel("Predicted Consumption (kWh/dia)", fontsize=12)
    plt.title("Predictions vs Reality", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    from utilities import metrics_calculator
    metrics  = metrics_calculator(y_true.flatten(), y_pred.flatten())

    text = f'R2 = {metrics["R2"]:.3f}\nMAE = {metrics["MAE"]:.2f} kWh'
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


    if save:
        plt.savefig("predictions_vs_reality.png", dpi=300, bbox_inches="tight")


    plt.show()

def plot_error_distribution(y_true, y_pred, save=False):
    
    mistakes = y_pred - y_true

    plt.figure(figsize=(10,6))


    plt.hist(mistakes, bins=30, edgecolor="black", alpha=0.7)
    plt.axvline(x=0, color="r", linestyle="--", linewidth=2, label="Error = 0")
    plt.axvline(x=np.mean(mistakes), color="g", linestyle="--", linewidth=2, label=f"Mean = {np.mean(mistakes):.2f}")


    plt.xlabel("Error (kWh)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Error Predict Distribution", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis="y")


    if save:
        plt.savefig("error_distribution.png", dpi=300, bbox_inches="tight")

    
    plt.show()


def plot_experiment_comparassion(results, save=False):

    names = [r["name"] for r in results]
    train_losses = [r["train_loss"] for r in results]
    val_losses = [r["val_loss"] for r in results]
    maes = [r["mae"] for r in results]


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    x = np.arange(len(names))
    width = 0.35


    ax1.bar(x-width/2, train_losses, width, label="Train  Loss", alpha=0.8)
    ax1.bar(x+width/2, val_losses, width, label="Val Loss", alpha=0.8)
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Loss Comparassion")
    ax1.set_xticks(x)
    ax1.set_xtickslabels(names, rotation=45, ha="righ")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")


    ax2.bar(names, maes, alpha=0.8, color="orange")
    ax2.set_ylabels("MAE (kWh)")
    ax2.set_title("Mean Absolute Error Comparassion")
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")


    plt.tight_layout()

    if save:
        plt.savefig("exmperiment_comparassion.png", dpi=300, bbox_inches="tight")


    plt.show()