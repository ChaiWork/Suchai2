# Pokémon TCG Battle Replay Viewer CLI

An interactive, responsive HTML replay viewer for the Kaggle Pokémon Company - PTCG AI Battle Challenge Simulation.

---

## ⚡ Features
- **Interactive Replay Controls:** Play, pause, step forward/backward, and slide through turns.
- **Card Zoom-in:** Tap on any card (in active, bench, or hand zones) to open a full-size modal view.
- **Card Image Extraction:** Automatically reads the competition PDF (`Card_ID List_JP.pdf`) and extracts cropped PNG images for card thumbnails.
- **Log Viewer:** Displays a scrollable list of recent game logs mapped dynamically to card names.
- **Generalization Support:** Handles custom or default decks dynamically.

---

## 📥 Setup & Requirements
To enable card image extraction from the official PDF, you need to install the dependencies:
```bash
pip install pymupdf pillow
```

---

## 🚀 How to Run

### 1. Compile and Open Replay (Automatic PDF Search)
The script will search your current directory and parent directories recursively for the competition PDF:
```bash
python battle_viewer/main.py path/to/replay.json
```

### 2. Specify PDF Path Manually
```bash
python battle_viewer/main.py path/to/replay.json --pdf path/to/Card_ID_List_JP.pdf
```

### 3. Compile Without Card Images (Fast/No PDF)
If you don't have the PDF or want to compile instantly:
```bash
python battle_viewer/main.py path/to/replay.json --no-images
```

---

## ⚙️ Options & Arguments
- `json_path`: (Required) Path to the battle replay JSON log file.
- `--pdf`: Path to the competition Card_ID List PDF.
- `--img-dir`: Directory to cache extracted card PNG images (default: `card_images`).
- `--output`: Output HTML filename (default: `battle_viewer.html`).
- `--no-images`: Compile HTML without extracting card images.
- `--no-open`: Do not automatically open the compiled viewer in the default browser.
