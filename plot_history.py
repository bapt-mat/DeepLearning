import h5py
import matplotlib.pyplot as plt

try:
    with h5py.File('training_history.h5', 'r') as f:
        epochs = f['epochs'][:]
        train_loss = f['train_loss'][:]
        val_loss = f['val_loss'][:]
        val_dice = f['val_dice'][:]
        val_iou = f['val_iou'][:]

    plt.figure(figsize=(14, 6))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
    plt.title('Baseline U-Net: Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (BCE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Metrics
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_dice, label='Dice Score', color='green', marker='s')
    plt.plot(epochs, val_iou, label='IoU Score', color='orange', marker='^')
    plt.title('Baseline U-Net: Segmentation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('baseline_curves.png')
    print("✅ Plot saved to baseline_curves.png")
    plt.show()

except FileNotFoundError:
    print("❌ File 'training_history.h5' not found. Download it from the cluster first!")