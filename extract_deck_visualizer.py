#!/usr/bin/env python3
"""
Pokémon TCG Card Image Extractor and Deck Visualizer Tool
-------------------------------------------------------
This utility extracts card images from the official PTCG competition PDF
based on a CSV deck list (e.g. deck.csv) and compiles them into a beautiful,
high-resolution composite deck visualizer image.

Features:
- Threaded GUI interface (removes UI freeze during PDF processing)
- Custom slate-dark theme matching professional developer interfaces
- Real-time progress updates and status logger
- Native file/folder dialog selectors
- Generates high-quality composite deck sheets with quantity badges
- Supports standard CSV lists (one Card ID per line or comma-separated quantity lists)
- Fallback CLI mode for headless execution or scripting
"""

import os
import sys
import csv
import argparse
import threading
import webbrowser
from pathlib import Path
from collections import Counter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Global constants for styling
BG_DARK = "#121824"      # Deep charcoal slate
PANEL_DARK = "#1e293b"   # Slate panel
TEXT_LIGHT = "#f1f5f9"   # White/off-white
TEXT_MUTED = "#94a3b8"   # Slate gray
ACCENT_BLUE = "#3b82f6"  # Premium modern blue
ACCENT_GREEN = "#10b981" # Emerald green
ACCENT_RED = "#ef4444"   # Red badge
BORDER_COLOR = "#334155" # Dark slate border

def open_file_or_folder(path: Path):
    """Platform-independent file opener."""
    try:
        if os.name == 'nt':
            os.startfile(str(path))
        elif sys.platform == 'darwin':
            import subprocess
            subprocess.Popen(['open', str(path)])
        else:
            import subprocess
            subprocess.Popen(['xdg-open', str(path)])
    except Exception as e:
        print(f"Error opening {path}: {e}")

# ==========================================
# CORE EXTRACTION AND COMPOSITION LOGIC
# ==========================================

def extract_card_image_page_map(pdf_path: Path) -> dict[int, int]:
    """
    Parses the internal links of Card_ID_List_*.pdf index pages
    to create a map of: Card ID -> Card Image Page Number (0-indexed).
    """
    try:
        import fitz
    except ImportError as e:
        raise ImportError("PyMuPDF is required. Install it using: pip install pymupdf") from e

    doc = fitz.open(str(pdf_path))
    mapping = {}
    ordered_targets = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        links = [l for l in page.get_links() if "page" in l and l.get("page") is not None]
        if not links:
            continue

        words = page.get_text("words")  # list of tuples: (x0, y0, x1, y1, "word", ...)
        links = sorted(links, key=lambda l: (float(l["from"].y0), float(l["from"].x0)))

        for link in links:
            rect = link["from"]
            y_mid = (float(rect.y0) + float(rect.y1)) / 2
            target = int(link["page"])
            ordered_targets.append(target)

            # Same line on the left side is considered the Card ID
            row_words = []
            for w in words:
                x0, y0, x1, y1, text = float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4])
                wy = (y0 + y1) / 2
                if x0 < 145 and abs(wy - y_mid) < 8 and text.strip().isdigit():
                    row_words.append((abs(wy - y_mid), x0, text.strip()))

            if row_words:
                row_words.sort()
                try:
                    card_id = int(row_words[0][2])
                    mapping[card_id] = target
                except Exception:
                    pass

    # Fallback to simple index ordering if PDF word extraction didn't align
    if len(mapping) < max(10, len(ordered_targets) // 2):
        fallback = {i + 1: p for i, p in enumerate(ordered_targets)}
        mapping.update({k: v for k, v in fallback.items() if k not in mapping})

    return mapping

def _crop_card_image(img):
    """
    Crops away the surrounding white margins from the extracted card PDF page image.
    """
    from PIL import Image, ImageChops

    bg = Image.new(img.mode, img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()

    if not bbox:
        return img

    left, top, right, bottom = bbox

    # Add a small padding (approx 4%) to avoid cutting card edges
    pad_x = max(8, int((right - left) * 0.04))
    pad_y = max(8, int((bottom - top) * 0.04))

    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(img.width, right + pad_x)
    bottom = min(img.height, bottom + pad_y)

    return img.crop((left, top, right, bottom))

def prepare_card_images(
    pdf_path: Path,
    out_dir: Path,
    card_ids: list[int],
    zoom: float = 2.0,
    overwrite: bool = False,
    progress_callback = None
) -> dict[int, Path]:
    """
    Renders card pages from PDF and crops them, saving each as a PNG image.
    Calls progress_callback(index, total_count, card_id) if provided.
    """
    try:
        import fitz
        from PIL import Image
    except ImportError as e:
        raise ImportError("PyMuPDF and Pillow are required. Install them: pip install pymupdf pillow") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    
    if progress_callback:
        progress_callback(0, 1, "Parsing PDF structure...")
        
    page_map = extract_card_image_page_map(pdf_path)
    target_ids = sorted({int(x) for x in card_ids if int(x) in page_map})
    total = len(target_ids)

    doc = fitz.open(str(pdf_path))
    saved = {}

    for idx, card_id in enumerate(target_ids):
        out_path = out_dir / f"{card_id}.png"
        
        if progress_callback:
            progress_callback(idx, total, f"Extracting Card {card_id} ({idx + 1}/{total})")
            
        if out_path.exists() and not overwrite:
            saved[card_id] = out_path
            continue

        page_no = page_map.get(card_id)
        if page_no is None:
            continue

        # Render PDF page to high-res image
        page = doc[page_no]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Crop borders and save
        img_cropped = _crop_card_image(img)
        img_cropped.save(out_path)
        saved[card_id] = out_path

    if progress_callback:
        progress_callback(total, total, "All card images extracted!")

    return saved

def generate_deck_grid(
    csv_path: Path,
    pdf_path: Path,
    output_dir: Path,
    output_image_path: Path,
    cols: int = 6,
    progress_callback = None
):
    """
    Full pipeline: reads CSV, extracts images, and builds a beautiful deck grid sheet.
    """
    from PIL import Image, ImageDraw, ImageFont

    if progress_callback:
        progress_callback(0, 100, "Reading CSV deck list...")

    # 1. Parse Card IDs from CSV
    card_ids = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Handle either a single ID list or standard comma-separated lines
            token = row[0].strip()
            if token.isdigit():
                # Check if there's a quantity column (e.g. ID,Quantity)
                if len(row) > 1 and row[1].strip().isdigit():
                    qty = int(row[1].strip())
                    card_ids.extend([int(token)] * qty)
                else:
                    card_ids.append(int(token))

    if not card_ids:
        raise ValueError("No valid Card IDs found in the selected CSV file.")

    total_cards = len(card_ids)
    counts = Counter(card_ids)
    unique_ids = sorted(list(counts.keys()))
    num_unique = len(unique_ids)

    # 2. Extract Card Images
    def item_callback(idx, total, msg):
        if progress_callback:
            # Map PDF extraction step to 10% - 90% range of global progress
            prog = 10 + int((idx / total) * 80)
            progress_callback(prog, 100, msg)

    saved_images = prepare_card_images(
        pdf_path=pdf_path,
        out_dir=output_dir,
        card_ids=unique_ids,
        progress_callback=item_callback
    )

    if progress_callback:
        progress_callback(90, 100, "Compiling composite deck sheet...")

    # 3. Canvas & Layout definitions
    card_w, card_h = 240, 236
    padding = 12
    header_h = 80
    footer_h = 30
    rows = (num_unique + cols - 1) // cols

    img_w = cols * card_w + (cols + 1) * padding
    img_h = header_h + rows * card_h + (rows + 1) * padding + footer_h

    # Create background image
    bg_color = (18, 24, 36) # Dark charcoal
    deck_img = Image.new("RGB", (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(deck_img)

    # Draw header panel
    header_color = (30, 41, 59) # Dark slate
    draw.rectangle([0, 0, img_w, header_h], fill=header_color)

    # Load system fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 26)
        subtitle_font = ImageFont.truetype("arial.ttf", 14)
        badge_font = ImageFont.truetype("arialbd.ttf", 15)
        footer_font = ImageFont.truetype("arial.ttf", 11)
    except IOError:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        badge_font = ImageFont.load_default()
        footer_font = ImageFont.load_default()

    # Draw title details
    draw.text((padding + 10, 14), "POKÉMON TCG DECK SHEET", fill=(255, 255, 255), font=title_font)
    draw.text((padding + 10, 48), f"Total Cards: {total_cards}  |  Unique Card Types: {num_unique}", fill=(148, 163, 184), font=subtitle_font)

    # Paste and format each card
    for idx, card_id in enumerate(unique_ids):
        r = idx // cols
        c = idx % cols

        x = padding + c * (card_w + padding)
        y = header_h + padding + r * (card_h + padding)

        card_file = output_dir / f"{card_id}.png"
        if not card_file.exists():
            continue

        try:
            with Image.open(card_file) as c_img:
                c_resized = c_img.resize((card_w, card_h), Image.Resampling.LANCZOS)
                deck_img.paste(c_resized, (x, y))

                # Draw elegant thin border
                draw.rectangle([x, y, x + card_w, y + card_h], outline=(51, 65, 85), width=1)

                # Quantity badge (rose-red for multiple, blue for single copy)
                qty = counts[card_id]
                badge_text = f"x{qty}"
                badge_bg = (225, 29, 72) if qty > 1 else (59, 130, 246)

                # Calculate text sizing
                if hasattr(draw, 'textbbox'):
                    l, t, r, b = draw.textbbox((0, 0), badge_text, font=badge_font)
                    text_w = r - l
                    text_h = b - t
                else:
                    text_w, text_h = draw.textsize(badge_text, font=badge_font)

                badge_w = max(42, text_w + 12)
                badge_h = text_h + 10
                bx0 = x + card_w - badge_w - 8
                by0 = y + card_h - badge_h - 8
                bx1 = x + card_w - 8
                by1 = y + card_h - 8

                # Draw badge outline and fill
                draw.rounded_rectangle([bx0, by0, bx1, by1], radius=6, fill=badge_bg, outline=(255, 255, 255), width=1)
                
                # Center text inside badge
                tx = bx0 + (badge_w - text_w) // 2
                ty = by0 + (badge_h - text_h) // 2 - 1
                draw.text((tx, ty), badge_text, fill=(255, 255, 255), font=badge_font)

        except Exception as e:
            print(f"Error drawing card {card_id} on canvas: {e}")

    # Draw footer
    draw.rectangle([0, img_h - footer_h, img_w, img_h], fill=(30, 41, 59))
    draw.text((padding + 10, img_h - footer_h + 8), "Generated dynamically from CSV deck list & Competition PDF", fill=(148, 163, 184), font=footer_font)

    # Save final sheet
    deck_img.save(output_image_path)
    if progress_callback:
        progress_callback(100, 100, "Successfully completed!")

# ==========================================
# DESKTOP INTERFACE (TKINTER GUI)
# ==========================================

class DeckVisualizerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Pokémon TCG Deck Visualizer")
        self.root.geometry("640x520")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)
        
        # Configure overall window theme styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure(
            "Horizontal.TProgressbar",
            troughcolor=BG_DARK,
            background=ACCENT_BLUE,
            bordercolor=BORDER_COLOR,
            lightcolor=ACCENT_BLUE,
            darkcolor=ACCENT_BLUE
        )

        self._create_widgets()
        self._find_default_files()

    def _create_widgets(self):
        # 1. Main Header
        header_frame = tk.Frame(self.root, bg=BG_DARK, pady=15)
        header_frame.pack(fill="x")
        
        lbl_title = tk.Label(
            header_frame,
            text="POKÉMON TCG DECK VISUALIZER",
            font=("Helvetica", 16, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_DARK
        )
        lbl_title.pack(anchor="w", padx=20)
        
        lbl_desc = tk.Label(
            header_frame,
            text="Extract card thumbnails from official PDF & generate a combined deck sheet image.",
            font=("Helvetica", 10),
            fg=TEXT_MUTED,
            bg=BG_DARK
        )
        lbl_desc.pack(anchor="w", padx=20, pady=(2, 0))

        # Divider line
        div = tk.Frame(self.root, height=1, bg=BORDER_COLOR)
        div.pack(fill="x", padx=20)

        # 2. Input Fields Section
        form_frame = tk.Frame(self.root, bg=BG_DARK, pady=15)
        form_frame.pack(fill="both", expand=True, padx=20)

        # Helper to create input rows
        def add_field_row(row_idx, label_text, default_val="", is_file=True, is_dir=False):
            lbl = tk.Label(
                form_frame,
                text=label_text,
                font=("Helvetica", 10, "bold"),
                fg=TEXT_LIGHT,
                bg=BG_DARK,
                anchor="w"
            )
            lbl.grid(row=row_idx, column=0, sticky="w", pady=10)

            entry = tk.Entry(
                form_frame,
                font=("Helvetica", 10),
                bg=PANEL_DARK,
                fg=TEXT_LIGHT,
                insertbackground=TEXT_LIGHT,
                highlightthickness=1,
                highlightbackground=BORDER_COLOR,
                highlightcolor=ACCENT_BLUE,
                bd=0
            )
            entry.grid(row=row_idx, column=1, sticky="ew", padx=(10, 5), ipady=5)
            entry.insert(0, default_val)

            def browse():
                if is_dir:
                    selected = filedialog.askdirectory(title=f"Select {label_text}")
                elif is_file:
                    file_types = [("PDF Files", "*.pdf")] if "PDF" in label_text else [("CSV Files", "*.csv"), ("All Files", "*.*")]
                    selected = filedialog.askopenfilename(title=f"Select {label_text}", filetypes=file_types)
                else:
                    file_types = [("PNG Files", "*.png")]
                    selected = filedialog.asksaveasfilename(title=f"Select {label_text}", filetypes=file_types, defaultextension=".png")
                
                if selected:
                    entry.delete(0, tk.END)
                    entry.insert(0, str(Path(selected).resolve()))

            btn_browse = tk.Button(
                form_frame,
                text="Browse...",
                font=("Helvetica", 9, "bold"),
                bg=PANEL_DARK,
                fg=TEXT_LIGHT,
                activebackground=BORDER_COLOR,
                activeforeground=TEXT_LIGHT,
                relief="flat",
                bd=0,
                padx=10,
                pady=4,
                cursor="hand2",
                command=browse
            )
            btn_browse.grid(row=row_idx, column=2, sticky="e", pady=10)
            
            btn_browse.bind("<Enter>", lambda e: btn_browse.config(bg=BORDER_COLOR))
            btn_browse.bind("<Leave>", lambda e: btn_browse.config(bg=PANEL_DARK))

            return entry

        form_frame.columnconfigure(1, weight=1)

        self.entry_csv = add_field_row(0, "Deck CSV File:", "deck.csv")
        self.entry_pdf = add_field_row(1, "Card ID List PDF:", "battle_viewer/Card_ID List_EN.pdf")
        self.entry_out_dir = add_field_row(2, "Output Card Folder:", "extracted_deck_images", is_file=False, is_dir=True)
        self.entry_composite = add_field_row(3, "Combined Image:", "deck_composite.png", is_file=False)

        # Extra options row (Columns count)
        lbl_cols = tk.Label(
            form_frame,
            text="Grid Columns:",
            font=("Helvetica", 10, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_DARK,
            anchor="w"
        )
        lbl_cols.grid(row=4, column=0, sticky="w", pady=10)

        self.entry_cols = tk.Entry(
            form_frame,
            font=("Helvetica", 10),
            bg=PANEL_DARK,
            fg=TEXT_LIGHT,
            insertbackground=TEXT_LIGHT,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=ACCENT_BLUE,
            bd=0,
            width=8
        )
        self.entry_cols.grid(row=4, column=1, sticky="w", padx=(10, 5), ipady=5)
        self.entry_cols.insert(0, "6")

        # Checkboxes for post-actions
        self.var_open_folder = tk.BooleanVar(value=True)
        self.chk_open_folder = tk.Checkbutton(
            form_frame,
            text="Open directory on success",
            variable=self.var_open_folder,
            bg=BG_DARK,
            fg=TEXT_MUTED,
            selectcolor=BG_DARK,
            activebackground=BG_DARK,
            activeforeground=TEXT_LIGHT,
            font=("Helvetica", 9),
            bd=0
        )
        self.chk_open_folder.grid(row=5, column=1, columnspan=2, sticky="w", pady=(5, 2))

        self.var_open_img = tk.BooleanVar(value=True)
        self.chk_open_img = tk.Checkbutton(
            form_frame,
            text="Open deck sheet image on success",
            variable=self.var_open_img,
            bg=BG_DARK,
            fg=TEXT_MUTED,
            selectcolor=BG_DARK,
            activebackground=BG_DARK,
            activeforeground=TEXT_LIGHT,
            font=("Helvetica", 9),
            bd=0
        )
        self.chk_open_img.grid(row=6, column=1, columnspan=2, sticky="w", pady=(2, 5))

        # Divider line
        div2 = tk.Frame(self.root, height=1, bg=BORDER_COLOR)
        div2.pack(fill="x", padx=20)

        # 3. Actions and Progress Section
        action_frame = tk.Frame(self.root, bg=BG_DARK, pady=15)
        action_frame.pack(fill="x", padx=20)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            action_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            style="Horizontal.TProgressbar"
        )
        self.progress_bar.pack(fill="x", pady=(0, 8))

        self.lbl_status = tk.Label(
            action_frame,
            text="Status: Ready",
            font=("Helvetica", 9, "italic"),
            fg=TEXT_MUTED,
            bg=BG_DARK,
            anchor="w"
        )
        self.lbl_status.pack(side="left")

        self.btn_run = tk.Button(
            action_frame,
            text="Extract & Generate Sheet",
            font=("Helvetica", 10, "bold"),
            bg=ACCENT_GREEN,
            fg=TEXT_LIGHT,
            activebackground="#059669",
            activeforeground=TEXT_LIGHT,
            relief="flat",
            bd=0,
            padx=18,
            pady=8,
            cursor="hand2",
            command=self.start_process
        )
        self.btn_run.pack(side="right")
        self.btn_run.bind("<Enter>", lambda e: self.btn_run.config(bg="#059669"))
        self.btn_run.bind("<Leave>", lambda e: self.btn_run.config(bg=ACCENT_GREEN))

    def _find_default_files(self):
        """Attempts to fill input fields with repository defaults if they exist."""
        cwd = Path(".").resolve()
        
        # Check deck.csv
        csv_file = cwd / "deck.csv"
        if csv_file.exists():
            self.entry_csv.delete(0, tk.END)
            self.entry_csv.insert(0, str(csv_file))

        # Check PDF paths in common locations
        pdf_file = cwd / "battle_viewer" / "Card_ID List_EN.pdf"
        if not pdf_file.exists():
            pdf_file = cwd / "battle_viewer" / "Card_ID List_JP.pdf"
        if not pdf_file.exists():
            # Quick search
            for match in cwd.rglob("*Card_ID*.pdf"):
                pdf_file = match
                break
        
        if pdf_file.exists():
            self.entry_pdf.delete(0, tk.END)
            self.entry_pdf.insert(0, str(pdf_file))

    def update_progress(self, percent: int, msg: str):
        """Thread-safe UI progress bar and status label updater."""
        self.progress_var.set(percent)
        self.lbl_status.config(text=f"Status: {msg}", fg=TEXT_LIGHT)
        self.root.update_idletasks()

    def start_process(self):
        """Triggered on button click. Launches extraction task on a background thread."""
        csv_path = Path(self.entry_csv.get().strip())
        pdf_path = Path(self.entry_pdf.get().strip())
        out_dir = Path(self.entry_out_dir.get().strip())
        comp_path = Path(self.entry_composite.get().strip())
        
        # Validate inputs
        if not csv_path.exists():
            messagebox.showerror("Error", f"Deck CSV file not found:\n{csv_path}")
            return
        if not pdf_path.exists():
            messagebox.showerror("Error", f"Competition PDF file not found:\n{pdf_path}")
            return
        
        try:
            cols = int(self.entry_cols.get().strip())
            if cols < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Grid Columns must be a positive integer.")
            return

        # Disable GUI elements to avoid duplicate threads
        self.btn_run.config(state="disabled", bg=PANEL_DARK, cursor="arrow")
        self.entry_csv.config(state="disabled")
        self.entry_pdf.config(state="disabled")
        self.entry_out_dir.config(state="disabled")
        self.entry_composite.config(state="disabled")
        self.entry_cols.config(state="disabled")

        # Launch background thread
        thread = threading.Thread(
            target=self.run_background_task,
            args=(csv_path, pdf_path, out_dir, comp_path, cols),
            daemon=True
        )
        thread.start()

    def run_background_task(self, csv_path: Path, pdf_path: Path, out_dir: Path, comp_path: Path, cols: int):
        """Handles PDF rendering and image construction in the background."""
        error_msg = None
        
        def local_progress(percent, total, status_text):
            # Convert status updates into 0-100 range
            norm_val = int((percent / total) * 100) if total > 0 else 0
            self.root.after(0, self.update_progress, norm_val, status_text)

        try:
            generate_deck_grid(
                csv_path=csv_path,
                pdf_path=pdf_path,
                output_dir=out_dir,
                output_image_path=comp_path,
                cols=cols,
                progress_callback=local_progress
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = str(e)

        # Callback on main thread to finish and restore GUI state
        self.root.after(0, self.finish_process, error_msg, out_dir, comp_path)

    def finish_process(self, error_msg: str | None, out_dir: Path, comp_path: Path):
        """Executed on the main thread after background task concludes."""
        # Re-enable inputs
        self.btn_run.config(state="normal", bg=ACCENT_GREEN, cursor="hand2")
        self.entry_csv.config(state="normal")
        self.entry_pdf.config(state="normal")
        self.entry_out_dir.config(state="normal")
        self.entry_composite.config(state="normal")
        self.entry_cols.config(state="normal")

        if error_msg:
            self.progress_var.set(0)
            self.lbl_status.config(text="Status: Failed", fg=ACCENT_RED)
            messagebox.showerror("Extraction Failed", f"An error occurred during extraction:\n{error_msg}")
        else:
            self.progress_var.set(100)
            self.lbl_status.config(text="Status: Success!", fg=ACCENT_GREEN)
            
            # Post success actions
            actions_taken = []
            if self.var_open_folder.get():
                open_file_or_folder(out_dir)
                actions_taken.append("opened output folder")
            if self.var_open_img.get():
                open_file_or_folder(comp_path)
                actions_taken.append("opened deck sheet image")
                
            suffix = f" ({', '.join(actions_taken)})" if actions_taken else ""
            messagebox.showinfo("Success", f"Deck Visualizer sheet generated successfully!{suffix}")

# ==========================================
# CLI MODE FALLBACK
# ==========================================

def run_cli(args):
    """Headless CLI wrapper execution."""
    csv_path = Path(args.csv)
    pdf_path = Path(args.pdf)
    out_dir = Path(args.out_dir)
    comp_path = Path(args.composite)
    cols = args.cols

    print("==========================================")
    print("Pokémon TCG Card Extractor (CLI Mode)")
    print(f"CSV Path:       {csv_path}")
    print(f"PDF Path:       {pdf_path}")
    print(f"Output Folder:  {out_dir}")
    print(f"Deck Sheet Path: {comp_path}")
    print(f"Columns:        {cols}")
    print("==========================================")

    if not csv_path.exists():
        print(f"Error: CSV path does not exist: {csv_path}")
        sys.exit(1)
    if not pdf_path.exists():
        print(f"Error: PDF path does not exist: {pdf_path}")
        sys.exit(1)

    def print_progress(current, total, status_text):
        pct = int((current / total) * 100) if total > 0 else 0
        sys.stdout.write(f"\rProgress: [{pct:3d}%] {status_text:<50}")
        sys.stdout.flush()
        if current == total and pct == 100:
            sys.stdout.write("\n")

    try:
        generate_deck_grid(
            csv_path=csv_path,
            pdf_path=pdf_path,
            output_dir=out_dir,
            output_image_path=comp_path,
            cols=cols,
            progress_callback=print_progress
        )
        print("Success! All operations completed successfully.")
    except Exception as e:
        print(f"\nError: Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ==========================================
# SCRIPT ENTRYPOINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Pokémon TCG Card Image Extractor and Deck Visualizer Tool.")
    parser.add_argument("--csv", type=str, default=None, help="Path to deck CSV file.")
    parser.add_argument("--pdf", type=str, default=None, help="Path to Card_ID List PDF.")
    parser.add_argument("--out-dir", type=str, default="extracted_deck_images", help="Output directory for individual card PNGs.")
    parser.add_argument("--composite", type=str, default="deck_composite.png", help="Output path for the compiled deck visualizer sheet.")
    parser.add_argument("--cols", type=int, default=6, help="Number of columns in the composite grid layout.")
    parser.add_argument("--cli", action="store_true", help="Force headless CLI mode (no Tkinter GUI launched).")

    args = parser.parse_args()

    if args.cli or args.csv or args.pdf:
        if not args.csv:
            args.csv = "deck.csv"
        if not args.pdf:
            default_pdf = Path("battle_viewer/Card_ID List_EN.pdf")
            if not default_pdf.exists():
                default_pdf = Path("battle_viewer/Card_ID List_JP.pdf")
            args.pdf = str(default_pdf)
            
        run_cli(args)
    else:
        root = tk.Tk()
        app = DeckVisualizerGUI(root)
        root.mainloop()

if __name__ == "__main__":
    main()
