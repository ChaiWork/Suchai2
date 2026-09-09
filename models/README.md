# Model Checkpoints

Model weights (`.pth` / `.pt` files) are **not committed** to this repository due to their size (~306 MB each).

## Trained Checkpoints

| File | Description | Size |
|---|---|---|
| `best_model.pth` | Highest evaluation win-rate checkpoint | ~306 MB |
| `latest_model.pth` | Final checkpoint from last training run | ~306 MB |

## Obtaining the Checkpoints

**Option 1 — Download the trained checkpoint (recommended):**

```bash
# Linux / macOS
wget https://github.com/ChaiWork/PTCG-AI/releases/download/v1.0.0/best_model.pth

# Windows PowerShell
Invoke-WebRequest -Uri "https://github.com/ChaiWork/PTCG-AI/releases/download/v1.0.0/best_model.pth" -OutFile "best_model.pth"
```

Or download manually from the [GitHub Releases page](https://github.com/ChaiWork/PTCG-AI/releases/latest).

**Option 2 — Train from scratch:**
```bash
python train.py --epochs 15 --self-play-episodes 100 --batch-size 256 --lr 5e-5 --eval-episodes 120 --num-workers 4
```
The `CheckpointManager` will automatically save `best_model.pth` to the project root when a new peak win-rate is reached.

**Option 2 — Download from the Kaggle Competition:**

Visit the [Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge) page for pre-trained weights if publicly released.

## Loading a Checkpoint

```python
import torch
from src.model import (
    MyModel, MODEL_D_MODEL, MODEL_NUM_HEADS,
    MODEL_D_FEEDFORWARD, MODEL_NUM_LAYERS_ENCODER, MODEL_NUM_LAYERS_DECODER,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel(
    MODEL_D_MODEL, MODEL_NUM_HEADS, MODEL_D_FEEDFORWARD,
    MODEL_NUM_LAYERS_ENCODER, MODEL_NUM_LAYERS_DECODER,
).to(device)

checkpoint = torch.load("best_model.pth", map_location=device, weights_only=True)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)
model.eval()
```

## Expected File Location

Place the checkpoint at the **project root**:
```
pokemon-tcg-ai/
├── best_model.pth   ← place here for inference / packaging
├── deck.csv
├── main.py
└── ...
```

The `package_submission.py` script will automatically pick up `best_model.pth` from the root when building the Kaggle submission archive.
