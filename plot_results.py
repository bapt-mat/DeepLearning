import h5py
import matplotlib.pyplot as plt
import glob

# Find all H5 files
files = glob.glob("results_*.h5")

plt.figure(figsize=(12, 6))

for file in files:
    name = file.replace("results_", "").replace(".h5", "")
    try:
        with h5py.File(file, 'r') as f:
            dice = f['val_dice'][:]
            plt.plot(dice, label=name)
            
        # Try to read score
        try:
            with open(f"score_{name}.txt", "r") as txt:
                score = float(txt.read().strip())
                print(f"{name}: oF1 = {score:.4f}")
        except: pass
    except: pass

plt.title("Ablation Studies: Validation Dice Curves")
plt.xlabel("Epochs")
plt.ylabel("Dice Score")
plt.legend()
plt.grid(True)
plt.savefig("final_comparison.png")
plt.show()