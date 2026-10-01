from glare import compute_glare
from data.dataloader_VerSe_new import VerSeDataLoader
from helper.arguments import get_parameter
from monai.transforms import Compose, NormalizeIntensity
import json
import os
import shutil
import re
from datetime import datetime
import torch
import gc
import psutil

# SYSTEM CLEANUP AND DIAGNOSTICS
def system_cleanup():
    """Clean up GPU memory and system resources."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Python Garbage Collection
    gc.collect()
    
    print("GPU memory and system cleaned up")
    
    # System-Diagnostics
    print(f" CPU Usage: {psutil.cpu_percent()}%")
    print(f" RAM Usage: {psutil.virtual_memory().percent}%")

# Single GPU setup
def setup_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU 0: {gpu_name}")
        print(f"   Optimized for stable performance")
        return True
    else:
        print("No GPU available - using CPU")
        return False


class ImageTransform:
    def __init__(self, mean, std):
        self.transforms = Compose([
            NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True),
        ])
    def __call__(self, data):
        return self.transforms(data)

# System cleanup 
system_cleanup()

# GPU check
gpu_available = setup_gpu()

# specs
spec_path = "/home/student/lisa_ma/results/VerSe_classifier/label_override_8ep_23092026_2135/spec.json"
with open(spec_path, "r") as f:
    spec = json.load(f)


spec_data = {
    "mean": [-111.3885],
    "std": [406.3665]
}


spec["dataset_dir"] = "/home/student/lisa_ma/prepared"
#spec["train_set_size"] = 0.01  
spec["batch_size"] = 1

#weights_dir = "/home/student/lisa_ma/results/VerSe_classifier/test_glare_27092025_1811/weights"
weights_dir = "/home/student/lisa_ma/results/VerSe_classifier/label_override_8ep_23092026_2135/weights"

# create results dir
results_base_dir = os.path.join("results", "glare")
os.makedirs(results_base_dir, exist_ok=True)

# Timestamp and run-info
now = datetime.now()
timestamp = now.strftime("%d-%m-%y_%H:%M")
#data_percent = int(spec["train_set_size"] * 100)
# date/time string safe for filenames (YYYYmmdd_HHMMSS)
run_ts = now.strftime("%Y%m%d_%H%M%S")

#print(f"=== GLARE-TEST ({data_percent}% Daten, Single GPU) ===")
#print(f"Use {data_percent}% of data")
print(f"Batch Size: {spec['batch_size']}")

# number of epochs 
all_files = os.listdir(weights_dir)
epoch_pattern = re.compile(r'^epoch_(\d+)$')
epoch_numbers = []

for filename in all_files:
    match = epoch_pattern.match(filename)
    if match:
        epoch_numbers.append(int(match.group(1)))

epoch_numbers.sort()
#last_x_epochs = epoch_numbers[-3:] if len(epoch_numbers) >= 3 else epoch_numbers
last_x_epochs = epoch_numbers # all epochs
num_epochs = len(last_x_epochs)

# create run folder
run_folder = f"{timestamp}_ep-{num_epochs}_single-gpu"
run_results_dir = os.path.join(results_base_dir, run_folder)
os.makedirs(run_results_dir, exist_ok=True)

print(f"Available epochs: {epoch_numbers}")
print(f"Using last {num_epochs} epochs: {last_x_epochs}")
print(f"Saving results to: {run_results_dir}")

# copy selected epochs to a temp folder for GLARE
temp_weights_dir = f"{weights_dir}_temp_test"
if os.path.exists(temp_weights_dir):
    shutil.rmtree(temp_weights_dir)
os.makedirs(temp_weights_dir)

for epoch_num in last_x_epochs:
    src = f"{weights_dir}/epoch_{epoch_num}"
    dst = f"{temp_weights_dir}/epoch_{epoch_num}"
    shutil.copy2(src, dst)
    print(f"  Copied: epoch_{epoch_num}")

# Dataloader set-up
data_module = VerSeDataLoader(
    spec=spec,
    train_transforms=ImageTransform(mean=spec_data["mean"], std=spec_data["std"]),
    val_transforms=ImageTransform(mean=spec_data["mean"], std=spec_data["std"]),
    num_workers=6
)
data_module.setup()
train_loader = data_module.train_dataloader()

# GPU memory status before start
if gpu_available:
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
    mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"GPU 0 memory: {mem_allocated:.1f}GB / {mem_total:.1f}GB used")
    print(f"GPU 0 Reserved: {mem_reserved:.1f}GB")

print(f"Analyzing {len(train_loader.dataset)} samples")


import time
start_time = time.time()

scores, grad_norms = compute_glare(
    dataloader=train_loader,
    num_classes=4,
    weights_dir=temp_weights_dir,
    spec=spec
)

end_time = time.time()
elapsed_time = end_time - start_time

# performance statistics
samples_per_second = len(scores) / elapsed_time
print(f"\nPerformance statistics:")
print(f"Total time: {elapsed_time/60:.1f} minutes")
print(f"Samples/second: {samples_per_second:.2f}")


# evaluate results
method = get_parameter(spec, "method", "glarex", str)
threshold = scores[f'{method} score'].quantile(get_parameter(spec, "threshold_fraction", 0.1, float))
mislabels = scores[scores[f'{method} score'] <= threshold]

print(f"\n=== GLARE RESULTS ===")
print(f"Analyzed samples: {len(scores)}")
print(f"Detected mislabels: {len(mislabels)} ({len(mislabels)/len(scores)*100:.1f}%)")
print(f"Threshold: {threshold:.6f}")

# safe results
mislabels_list_path = os.path.join(run_results_dir, f'mislabels_list_{run_ts}.csv')
mislabels_only_path = os.path.join(run_results_dir, f'mislabels_only_{run_ts}.csv')

scores_pkl_path = os.path.join(run_results_dir, f'scores_{run_ts}.pkl')
grad_norms_pkl_path = os.path.join(run_results_dir, f'grad_norms_{run_ts}.pkl')

scores.to_csv(mislabels_list_path, index=False)
mislabels.to_csv(mislabels_only_path, index=False)

scores.to_pickle(scores_pkl_path)
grad_norms.to_pickle(grad_norms_pkl_path)

print(f"Additionally saved:")
print(f"   {os.path.basename(scores_pkl_path)} (GLARE scores)")
print(f"   {os.path.basename(grad_norms_pkl_path)} (raw gradient norms)")


run_info = {
    "timestamp": now.isoformat(),
    "run_ts": run_ts,
    "spec_path": spec_path,
    "weights_dir": weights_dir,
    "weights_model": os.path.basename(os.path.dirname(weights_dir)),
    "gpu_name": torch.cuda.get_device_name(0) if gpu_available else "CPU",
    "batch_size": spec["batch_size"],
    #"data_percent": data_percent,
    "processing_time_minutes": elapsed_time / 60,
    "samples_per_second": samples_per_second,
    "total_samples": len(train_loader.dataset),
    "analyzed_samples": len(scores),
    "identified_mislabels": len(mislabels),
    "num_workers": 6,
    "system_cleanup": True,
    "performance_optimized": True
}

with open(os.path.join(run_results_dir, 'run_info.json'), 'w') as f:
    json.dump(run_info, f, indent=2)

print(f"\nGLARE run complete.")
print(f"Results saved to: {run_results_dir}")

# Final Cleanup
shutil.rmtree(temp_weights_dir)
print(f"Temporary directory removed.")

if gpu_available:
    torch.cuda.empty_cache()
    print("GPU memory cleared.")