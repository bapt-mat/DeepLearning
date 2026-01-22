import pandas as pd
import numpy as np
from kaggle_metric import score

# --- SCENARIO: An image with 2 separate forged regions ---
# We simulate the RLE strings directly to avoid dependency on image loading
# Instance 1: Pixels 0-2 are white
# Instance 2: Pixels 10-12 are white
gt_rle = "[[1, 3]];[[11, 3]]" 
shape = "[10, 10]" # 10x10 image

# 1. Perfect Prediction (Should be 1.0)
# We predict exactly the same two instances
pred_perfect = "[[1, 3]];[[11, 3]]"

# 2. "Merged" Prediction (Should be LOW, < 1.0)
# We predict one big blob covering both (Merging instances is bad for oF1)
pred_merged = "[[1, 13]]" 

# 3. "Over" Prediction (Should be Penalized)
# We predict the 2 real instances PLUS a 3rd ghost instance
pred_excess = "[[1, 3]];[[11, 3]];[[20, 3]]"

# 4. Missed Prediction (Should be 0.0)
pred_miss = "authentic"

def test_scenario(name, pred_rle, expected_range):
    # Construct DataFrames like the evaluation script does
    sol = pd.DataFrame({'row_id': [0], 'annotation': [gt_rle], 'shape': [shape]})
    sub = pd.DataFrame({'row_id': [0], 'annotation': [pred_rle]})
    
    result = score(sol, sub, 'row_id')
    print(f"🧪 {name}: Score = {result:.4f} (Expected: {expected_range})")

print("--- 🧐 Verifying oF1 Metric Logic ---")
test_scenario("Perfect Match", pred_perfect, "1.0")
test_scenario("Merged Instances", pred_merged, "< 1.0 (Poor overlap)")
test_scenario("Excess Prediction", pred_excess, "< 1.0 (Penalty applied)")
test_scenario("Complete Miss", pred_miss, "0.0")