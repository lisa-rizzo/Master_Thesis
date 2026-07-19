import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# Pfad zur DataLoader-Implementierung im Repo
from dataloaders.dataloader_VerSe import VerSeDataLoader

# Standard mean/std aus deinem train_VerSe_densenet.py
DEFAULT_MEAN = [-111.3885]
DEFAULT_STD = [406.3665]

def make_transform(mean, std):
    """
    Versucht MONAI-Transforms zu verwenden; falls MONAI nicht installiert ist,
    liefert ein einfaches Normalisierungs-Callable als Fallback.
    """
    try:
        from monai.transforms import Compose, NormalizeIntensity
        return Compose([NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True)])
    except Exception:
        def _simple_norm(data):
            # Falls DataLoader dict mit 'img' liefert: normalisiere inplace
            if isinstance(data, dict) and 'img' in data:
                arr = data['img']
                # arr kann numpy oder torch sein
                if isinstance(arr, np.ndarray):
                    data['img'] = (arr - mean[0]) / (std[0] if std[0] != 0 else 1.0)
                elif isinstance(arr, torch.Tensor):
                    data['img'] = (arr - mean[0]) / (std[0] if std[0] != 0 else 1.0)
            return data
        return _simple_norm

def inspect_batch(batch):
    """
    Erwartet das Format (imgs, labels, ids) oder (imgs, labels).
    Druckt Formate und Beispiel-IDs.
    """
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        imgs = batch[0]
        labels = batch[1]
        ids = batch[2] if len(batch) > 2 else None
    else:
        print("Unbekanntes Batch-Format:", type(batch))
        return

    print("imgs type:", type(imgs))
    try:
        # Tensor oder numpy
        if hasattr(imgs, "shape"):
            print("imgs shape:", imgs.shape)
    except Exception:
        pass

    print("labels type:", type(labels))
    try:
        # falls torch tensor -> numpy for nicer print
        if isinstance(labels, torch.Tensor):
            print("labels sample:", labels[:8].cpu().numpy())
        else:
            print("labels sample:", labels[:8])
    except Exception:
        pass

    if ids is not None:
        print("IDs (erste 10):")
        try:
            # Handle if ids is a tensor or list
            if isinstance(ids, (list, tuple)):
                for i, idv in enumerate(ids[:10]):
                    print(f"  [{i}] {idv}")
            elif isinstance(ids, torch.Tensor):
                print("  (ids ist ein Tensor) sample:", ids[:10])
            else:
                # fallback
                for i, idv in enumerate(list(ids)[:10]):
                    print(f"  [{i}] {idv}")
        except Exception:
            print("  (konnte IDs nicht iterieren, raw repr:) ", repr(ids))

    # Inspect first image min/max
    try:
        first_img = imgs[0]
        if isinstance(first_img, torch.Tensor):
            print("first_img dtype/min/max:", first_img.dtype, float(first_img.min()), float(first_img.max()))
        elif isinstance(first_img, np.ndarray):
            print("first_img dtype/min/max:", first_img.dtype, float(first_img.min()), float(first_img.max()))
    except Exception as e:
        print("Konnte first_img stats nicht ermitteln:", e)

def save_slice_as_png(img, outpath="sample_slice.png"):
    """
    Erwartet img als torch.Tensor oder numpy array mit Form (C,Z,Y,X) oder (Z,Y,X).
    Speichert mittlere Z-Slice als PNG (skaliert).
    """
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.array(img)

    # Handle shape (C,Z,Y,X) or (Z,Y,X)
    if arr.ndim == 4:
        # C,Z,Y,X -> take channel 0
        arr = arr[0]
    if arr.ndim == 3:
        z = arr.shape[0] // 2
        slice_ = arr[z]
    elif arr.ndim == 2:
        slice_ = arr
    else:
        raise ValueError(f"Unerwartete img ndim: {arr.ndim}")

    # Normalize for display
    mn, mx = np.nanmin(slice_), np.nanmax(slice_)
    if mx - mn > 0:
        slice_disp = (slice_ - mn) / (mx - mn)
    else:
        slice_disp = np.zeros_like(slice_)
    plt.imsave(outpath, slice_disp, cmap="gray")
    print(f"Saved slice to {outpath}")

def is_nontrivial_mapping(mapping):
    """
    Prüft, ob eine pid_corrections-Mapping nicht identisch ist.
    mapping: dict(original_vert -> corrected_vert)
    """
    if not isinstance(mapping, dict):
        return True
    for k, v in mapping.items():
        try:
            if int(k) != int(v):
                return True
        except Exception:
            # Wenn keys/values nicht int-konvertierbar sind - betrachte als nontrivial
            if k != v:
                return True
    return False

def load_pid_json(path):
    """
    Erwartet eine JSON-Datei mit Struktur z.B.:
    {
      "pid_fix_1820": ["PID1", "PID2"],
      "pid_fix_28": ["PID3"]
    }
    Liefert dict mit keys ggf. leer.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PID JSON not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    # Normalize keys
    out = {}
    out["pid_fix_1820"] = list(data.get("pid_fix_1820", [])) if isinstance(data, dict) else []
    out["pid_fix_28"] = list(data.get("pid_fix_28", [])) if isinstance(data, dict) else []
    return out

def main():
    parser = argparse.ArgumentParser(description="Quick test for VerSeDataLoader")
    parser.add_argument("--dataset_dir", type=str, default="/home/student/lisa_ma/prepared",
                        help="Pfad zum prepared-Ordner (oder spezifischem dataset-Ordner)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size für Test")
    parser.add_argument("--num_workers", type=int, default=2, help="Num workers für DataLoader")
    parser.add_argument("--train_fraction", type=float, default=0.02,
                        help="Kleiner Bruchteil des Trainingssets zum schnellen Smoke-Test (0..1).")
    parser.add_argument("--save_slice", action="store_true", help="Speichere eine Z-Slice PNG der ersten Sample im Batch")
    parser.add_argument("--pid_json", type=str, default=None,
                        help="Optional: Pfad zu JSON mit pid_fix_1820 / pid_fix_28 Listen (siehe README unten).")
    args = parser.parse_args()

    # Minimal-spec; bei Bedarf kannst du weitere keys hinzufügen (z.B. pid_fix_1820)
    spec = {
        "dataset_dir": args.dataset_dir,
        "directory": "results/VerSe_classifier_test",
        "job_name": "dataloader_smoke",
        "batch_size": args.batch_size,
        "holdout_set_size": 0,
        "use_train_for_val": False,
        "train_set_size": args.train_fraction,  # nutze nur einen kleinen Bruchteil für Test
    }

    # Load PID lists from JSON if provided and attach to spec
    if args.pid_json:
        try:
            pid_data = load_pid_json(args.pid_json)
            # attach lists only if non-empty
            if pid_data.get("pid_fix_1820"):
                spec["pid_fix_1820"] = pid_data["pid_fix_1820"]
            if pid_data.get("pid_fix_28"):
                spec["pid_fix_28"] = pid_data["pid_fix_28"]
            print("Loaded PID JSON. keys set in spec:",
                  [k for k in ("pid_fix_1820","pid_fix_28") if k in spec])
        except Exception as e:
            print("Fehler beim Laden der PID JSON:", e)
            return

    print("Spec (kurz):", json.dumps(spec, indent=2))

    transform = make_transform(DEFAULT_MEAN, DEFAULT_STD)

    print("Instantiating VerSeDataLoader...")
    dm = VerSeDataLoader(spec=spec, train_transforms=transform, val_transforms=transform, num_workers=args.num_workers)

    print("Calling setup() ...")
    dm.setup()

    # --- Safety: ensure pid_corrections exists on dataset instances BEFORE creating DataLoader workers ---
    # Wenn dieses Attribut erst später gesetzt wird, bekommen Worker einen AttributeError beim Zugriff.
    if not hasattr(dm, "train_dataset") or dm.train_dataset is None:
        raise RuntimeError("dm.train_dataset not found after setup()")
    if not hasattr(dm.train_dataset, "pid_corrections") or dm.train_dataset.pid_corrections is None:
        dm.train_dataset.pid_corrections = {}
    if hasattr(dm, "val_dataset") and (not hasattr(dm.val_dataset, "pid_corrections") or dm.val_dataset.pid_corrections is None):
        dm.val_dataset.pid_corrections = {}

    # Versuche verschiedene Weisen, die Dataset-Größe abzulesen
    def try_len(obj, name):
        try:
            l = len(obj)
            print(f"{name} length: {l}")
        except Exception:
            print(f"{name} has no len() or couldn't be measured.")

    # Mögliche Attribute: train_dataset, val_dataset, train_dataloader, val_dataloader
    try_len(getattr(dm, "train_dataset", None), "dm.train_dataset")
    try_len(getattr(dm, "val_dataset", None), "dm.val_dataset")

    try:
        train_loader = dm.train_dataloader()
        print("Train dataloader created:", train_loader)
    except Exception as e:
        print("Fehler beim Erstellen train_dataloader():", e)
        return

    # Hole eine Batch
    try:
        batch_iter = iter(train_loader)
        batch = next(batch_iter)
        print("Got a batch from train_dataloader()")
        inspect_batch(batch)
        if args.save_slice:
            # extrahiere first img und speichere mittlere Z-Slice
            imgs = batch[0]
            save_slice_as_png(imgs[0], outpath="dataloader_sample_slice.png")
    except StopIteration:
        print("Train loader lieferte keine Batches (leerer Loader).")
    except Exception as e:
        print("Fehler beim Lesen einer Batch:", e)

    # --- Ausgabe der PIDs, die Korrekturen haben (non-trivial mappings) ---
    print("\n--- PIDs mit angewendeten Korrekturen (non-trivial pid_corrections) ---")
    ds = dm.train_dataset
    pid_corr = getattr(ds, "pid_corrections", {}) or {}
    if not pid_corr:
        print("Keine pid_corrections gefunden (leeres Mapping).")
    else:
        non_trivial = []
        for pid, mapping in pid_corr.items():
            if is_nontrivial_mapping(mapping):
                non_trivial.append((pid, mapping))
        if not non_trivial:
            print("Keine nicht-trivialen Korrekturen gefunden (alle Mappings sind identisch).")
        else:
            print(f"Anzahl PIDs mit echten Korrekturen: {len(non_trivial)}")
            # Ausgabe aller oder ersten 200
            for i, (pid, mapping) in enumerate(non_trivial):
                if i >= 200:
                    print(f"... (noch {len(non_trivial)-200} weitere PIDs nicht angezeigt)")
                    break
                print(f"\nPID: {pid}")
                try:
                    for k in sorted(mapping.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
                        print(f"  {k} -> {mapping[k]}")
                except Exception:
                    print("  (mapping):", mapping)

if __name__ == "__main__":
    main()