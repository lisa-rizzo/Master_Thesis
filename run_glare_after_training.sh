#!/bin/bash
# Waits for the latest classifier training run to finish (epoch_14 checkpoint present),
# then updates run_glare.py and starts the GLARE scoring run.

CLASSIFIER_DIR="/home/student/lisa_ma/results/VerSe_classifier"
RUN_GLARE="/home/student/lisa_ma/run_glare.py"
CHECK_INTERVAL=120  # seconds between checks

echo "[$(date)] Waiting for training to finish (watching for epoch_3 checkpoint)..."

while true; do
    # Find the most recently modified classifier run directory
    LATEST=$(ls -td "$CLASSIFIER_DIR"/*/weights 2>/dev/null | head -1 | xargs dirname)

    if [ -n "$LATEST" ] && [ -f "$LATEST/weights/epoch_3" ]; then
        echo "[$(date)] Found epoch_3 in $LATEST — training complete!"
        break
    fi

    echo "[$(date)] Not done yet (latest: ${LATEST:-none}). Checking again in ${CHECK_INTERVAL}s..."
    sleep $CHECK_INTERVAL
done

# Update spec_path in run_glare.py
NEW_SPEC="$LATEST/spec.json"
echo "[$(date)] Updating run_glare.py spec_path to: $NEW_SPEC"
sed -i "s|spec_path = .*|spec_path = \"$NEW_SPEC\"|" "$RUN_GLARE"

# Start GLARE
echo "[$(date)] Starting GLARE run..."
source /home/student/miniconda3/etc/profile.d/conda.sh
conda activate mislabeldet
cd /home/student/lisa_ma
python3 run_glare.py
echo "[$(date)] GLARE run finished."
