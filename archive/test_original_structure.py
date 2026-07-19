# Test ob original glare.py das Problem hat
import sys
import os
sys.path.append('/home/student/lisa_ma')

import pandas as pd
import numpy as np

def test_original_glare_structure():
    # Simuliere das DataFrame das aus glare.py kommen würde
    data = []
    
    batch_size = 2
    num_classes = 3
    num_epochs = 2
    
    # Simuliere was wirklich aus glare.py kommt
    row_id = 0
    for batch in range(1):  # Ein Batch
        for epoch in range(num_epochs):
            for c in range(num_classes):
                for sample in range(batch_size):
                    if c == 0:
                        # Bei c=0: eine Zeile mit allen Informationen
                        data.append({
                            'id': f'sample_{batch}_{sample}',
                            'epoch': epoch,
                            'label': sample % 2,
                            'c0_grad_norm': 1.0 + row_id,
                            'c1_grad_norm': 2.0 + row_id,
                            'c2_grad_norm': 3.0 + row_id
                        })
                        row_id += 1
                    # Bei c > 0 passiert NICHTS (nur grad_norm wird zur existierenden Zeile hinzugefügt)
    
    df = pd.DataFrame(data)
    print("Simuliertes DataFrame aus glare.py:")
    print(df)
    print(f"\nShape: {df.shape}")
    
    # Test Pivot
    try:
        pivot = df.pivot(index="id", columns="epoch", values="c0_grad_norm")
        print("\nPivot erfolgreich:")
        print(pivot)
    except Exception as e:
        print(f"\nPivot Fehler: {e}")

if __name__ == "__main__":
    test_original_glare_structure()