import h5py
import matplotlib.pyplot as plt
import os

# Define the 3 specific studies you need for the report
studies = {
    "1_Pretraining_Effect": {
        "title": "Effect of Pre-training (ResNet-34)",
        "models": {
            "unet_baseline": {"label": "ImageNet Weights", "color": "green", "style": "-"},
            "unet_scratch":  {"label": "Random Init (Scratch)", "color": "red", "style": "--"}
        }
    },
    "2_Architecture_Compare": {
        "title": "CNN (U-Net) vs Transformer (SegFormer)",
        "models": {
            "unet_baseline":      {"label": "U-Net (ResNet34)", "color": "blue", "style": "-"},
            "segformer_baseline": {"label": "SegFormer (B0)", "color": "orange", "style": "-"}
        }
    },
    "3_Loss_Function": {
        "title": "BCE vs Dice Loss (U-Net)",
        "models": {
            "unet_baseline": {"label": "BCE Loss", "color": "purple", "style": "-"},
            "unet_dice":     {"label": "Dice Loss", "color": "cyan", "style": "--"}
        }
    }
}

for study_name, config in studies.items():
    plt.figure(figsize=(10, 6))
    
    for filename, style in config["models"].items():
        fname = f"results_{filename}.h5"
        if not os.path.exists(fname):
            print(f"⚠️ Missing file: {fname} (Skipping)")
            continue
            
        try:
            with h5py.File(fname, 'r') as f:
                # Load Dice scores
                dice = f['val_dice'][:]
                epochs = f['epochs'][:]
                
                # Plot
                plt.plot(epochs, dice, 
                         label=style["label"], 
                         color=style["color"], 
                         linestyle=style["style"], 
                         linewidth=2)
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    plt.title(config["title"], fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Validation Dice Score", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Save plot
    save_path = f"plot_{study_name}.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved plot: {save_path}")
    plt.close()