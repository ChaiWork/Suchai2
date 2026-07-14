#!/usr/bin/env python3
"""
Pokémon TCG Battle Replay Viewer Compiler
----------------------------------------
This utility compiles Kaggle Pokémon Company PTCG AI Battle Challenge replay logs
into an interactive, responsive HTML page to view game progression, logs, and card actions.

Usage:
1. Double-click or run 'python tcg_replay_viewer.py' to launch the desktop GUI.
2. Select a replay JSON, choose the competition PDF for image extraction, and launch.
3. Headless CLI mode is available by passing arguments (e.g. --json replay.json).
"""

import os
import sys
import argparse
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Global constants for styling
BG_DARK = "#121824"      # Deep charcoal slate
PANEL_DARK = "#1e293b"   # Slate panel
TEXT_LIGHT = "#f1f5f9"   # White/off-white
TEXT_MUTED = "#94a3b8"   # Slate gray
ACCENT_BLUE = "#3b82f6"  # Premium modern blue
ACCENT_GREEN = "#10b981" # Emerald green
ACCENT_RED = "#ef4444"   # Red
BORDER_COLOR = "#334155" # Dark slate border

# Add script directory to path so we can find battle_viewer
SCRIPT_DIR = Path(__file__).resolve().parent
BATTLE_VIEWER_DIR = SCRIPT_DIR / "battle_viewer"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from battle_viewer import viewer
from battle_viewer.main import find_default_pdf

# ==========================================
# DESKTOP INTERFACE (TKINTER GUI)
# ==========================================

class ReplayViewerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Pokémon TCG Replay Viewer")
        self.root.geometry("640x440")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        self._create_widgets()
        self._find_default_files()

    def _create_widgets(self):
        # 1. Main Header
        header_frame = tk.Frame(self.root, bg=BG_DARK, pady=15)
        header_frame.pack(fill="x")
        
        lbl_title = tk.Label(
            header_frame,
            text="POKÉMON TCG REPLAY COMPILER",
            font=("Helvetica", 16, "bold"),
            fg=TEXT_LIGHT,
            bg=BG_DARK
        )
        lbl_title.pack(anchor="w", padx=20)
        
        lbl_desc = tk.Label(
            header_frame,
            text="Compile battle replay JSON logs into an interactive, responsive HTML page.",
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
        def add_field_row(row_idx, label_text, default_val="", is_file=True, file_type_desc="JSON"):
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
                if is_file:
                    if file_type_desc == "JSON":
                        file_types = [("JSON Files", "*.json"), ("All Files", "*.*")]
                    elif file_type_desc == "PDF":
                        file_types = [("PDF Files", "*.pdf")]
                    else:
                        file_types = [("HTML Files", "*.html")]
                        
                    selected = filedialog.askopenfilename(title=f"Select {label_text}", filetypes=file_types)
                else:
                    file_types = [("HTML Files", "*.html")]
                    selected = filedialog.asksaveasfilename(title=f"Select {label_text}", filetypes=file_types, defaultextension=".html")
                
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

        self.entry_json = add_field_row(0, "Replay JSON File:", "", is_file=True, file_type_desc="JSON")
        self.entry_pdf = add_field_row(1, "Card ID List PDF:", "", is_file=True, file_type_desc="PDF")
        self.entry_output = add_field_row(2, "Output HTML File:", "battle_viewer.html", is_file=False)

        # Checkboxes for configurations
        self.var_prepare_images = tk.BooleanVar(value=True)
        self.chk_prepare_images = tk.Checkbutton(
            form_frame,
            text="Extract card thumbnail images from PDF",
            variable=self.var_prepare_images,
            bg=BG_DARK,
            fg=TEXT_MUTED,
            selectcolor=BG_DARK,
            activebackground=BG_DARK,
            activeforeground=TEXT_LIGHT,
            font=("Helvetica", 9),
            bd=0
        )
        self.chk_prepare_images.grid(row=3, column=1, columnspan=2, sticky="w", pady=(5, 2))

        self.var_open_browser = tk.BooleanVar(value=True)
        self.chk_open_browser = tk.Checkbutton(
            form_frame,
            text="Open HTML viewer in browser immediately",
            variable=self.var_open_browser,
            bg=BG_DARK,
            fg=TEXT_MUTED,
            selectcolor=BG_DARK,
            activebackground=BG_DARK,
            activeforeground=TEXT_LIGHT,
            font=("Helvetica", 9),
            bd=0
        )
        self.chk_open_browser.grid(row=4, column=1, columnspan=2, sticky="w", pady=(2, 5))

        # Divider line
        div2 = tk.Frame(self.root, height=1, bg=BORDER_COLOR)
        div2.pack(fill="x", padx=20)

        # 3. Actions Section
        action_frame = tk.Frame(self.root, bg=BG_DARK, pady=15)
        action_frame.pack(fill="x", padx=20)

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
            text="Compile & Launch",
            font=("Helvetica", 10, "bold"),
            bg=ACCENT_GREEN,
            fg=TEXT_LIGHT,
            activebackground="#059669",
            activeforeground=TEXT_LIGHT,
            relief="flat",
            bd=0,
            padx=20,
            pady=8,
            cursor="hand2",
            command=self.compile_replay
        )
        self.btn_run.pack(side="right")
        self.btn_run.bind("<Enter>", lambda e: self.btn_run.config(bg="#059669"))
        self.btn_run.bind("<Leave>", lambda e: self.btn_run.config(bg=ACCENT_GREEN))

    def _find_default_files(self):
        """Pre-populates forms using project defaults."""
        pdf_path = find_default_pdf()
        if pdf_path:
            self.entry_pdf.delete(0, tk.END)
            self.entry_pdf.insert(0, str(pdf_path.resolve()))
            
        # Try to locate any .json files in the battle_viewer directory
        battle_viewer_logs = list(BATTLE_VIEWER_DIR.glob("*.json"))
        if battle_viewer_logs:
            self.entry_json.delete(0, tk.END)
            self.entry_json.insert(0, str(battle_viewer_logs[0].resolve()))

    def compile_replay(self):
        json_path = Path(self.entry_json.get().strip())
        pdf_path = Path(self.entry_pdf.get().strip())
        out_html = Path(self.entry_output.get().strip())
        
        # Validation checks
        if not json_path.exists():
            messagebox.showerror("Error", f"Replay JSON log file not found:\n{json_path}")
            return
            
        prepare_images = self.var_prepare_images.get()
        if prepare_images:
            if not pdf_path.exists():
                # Ask if they want to compile without images
                res = messagebox.askyesno(
                    "Warning",
                    f"Card ID List PDF not found at:\n{pdf_path}\n\nDo you want to compile the replay without card images?"
                )
                if res:
                    prepare_images = False
                else:
                    return

        self.btn_run.config(state="disabled", bg=PANEL_DARK, cursor="arrow")
        self.lbl_status.config(text="Status: Compiling replay...", fg=TEXT_LIGHT)
        self.root.update_idletasks()

        try:
            out = viewer.write_html(
                json_path=json_path,
                out_html=out_html,
                card_image_dir="battle_viewer/card_images",
                card_pdf=pdf_path if prepare_images else None,
                prepare_images=prepare_images,
                overwrite_images=False
            )
            
            self.lbl_status.config(text="Status: Success!", fg=ACCENT_GREEN)
            
            actions = ["Compiled HTML saved."]
            if self.var_open_browser.get():
                file_url = f"file:///{str(out.resolve()).replace(os.sep, '/')}"
                webbrowser.open(file_url)
                actions.append("Opened in browser.")
                
            messagebox.showinfo("Success", f"Replay compiled successfully!\n{', '.join(actions)}")
        except Exception as e:
            self.lbl_status.config(text="Status: Failed", fg=ACCENT_RED)
            messagebox.showerror("Compilation Failed", f"An error occurred:\n{str(e)}")
        finally:
            self.btn_run.config(state="normal", bg=ACCENT_GREEN, cursor="hand2")

# ==========================================
# CLI MODE COMPILATION
# ==========================================

def run_cli(args):
    json_path = Path(args.json)
    out_html = Path(args.output)
    pdf_path = Path(args.pdf) if args.pdf else None
    
    print("==========================================")
    print("Pokémon TCG Replay Viewer Compiler")
    print(f"Replay JSON:    {json_path}")
    print(f"Output HTML:    {out_html}")
    print(f"Card PDF:       {pdf_path}")
    print(f"Extract Images: {not args.no_images}")
    print(f"Auto-Open:      {not args.no_open}")
    print("==========================================")
    
    if not json_path.exists():
        print(f"Error: JSON path does not exist: {json_path}")
        sys.exit(1)
        
    prepare_images = not args.no_images
    if prepare_images:
        if not pdf_path:
            pdf_path = find_default_pdf()
            
        if not pdf_path or not pdf_path.exists():
            print("Warning: Could not find competition Card ID PDF automatically.")
            print("Replay will be compiled without extracting card image thumbnails.")
            prepare_images = False
        else:
            print(f"Using competition PDF: {pdf_path}")
            
    print("Compiling battle replay...")
    try:
        out = viewer.write_html(
            json_path=json_path,
            out_html=out_html,
            card_image_dir="battle_viewer/card_images",
            card_pdf=pdf_path,
            prepare_images=prepare_images,
            overwrite_images=False
        )
        print(f"Success! Compiled HTML viewer saved to: {out.resolve()}")
        
        if not args.no_open:
            file_url = f"file:///{str(out.resolve()).replace(os.sep, '/')}"
            print(f"Opening viewer in browser: {file_url}")
            webbrowser.open(file_url)
            
    except Exception as e:
        print(f"Error compiling replay: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ==========================================
# SCRIPT ENTRYPOINT
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Pokémon TCG Battle Replay HTML Compiler.")
    parser.add_argument("--json", type=str, default=None, help="Path to the battle replay JSON log file.")
    parser.add_argument("--pdf", type=str, default=None, help="Path to the competition Card_ID List PDF. Searches automatically if omitted.")
    parser.add_argument("--output", type=str, default="battle_viewer.html", help="Output HTML filename (default: battle_viewer.html).")
    parser.add_argument("--no-images", action="store_true", help="Compile HTML without extracting or loading card images.")
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open the compiled viewer in the default browser.")
    parser.add_argument("--cli", action="store_true", help="Force headless CLI mode (no Tkinter GUI launched).")

    args = parser.parse_args()

    # Switch to CLI mode if any arguments (json or cli) are passed
    if args.cli or args.json:
        # Fallback for json argument in case it was omitted but CLI was forced
        if not args.json:
            # Look for any json log file in battle_viewer directory
            logs = list(BATTLE_VIEWER_DIR.glob("*.json"))
            if logs:
                args.json = str(logs[0])
                print(f"Auto-selected battle log: {args.json}")
            else:
                print("Error: CLI mode requires specifying a JSON path using --json <path>.")
                sys.exit(1)
        run_cli(args)
    else:
        # Run GUI mode
        root = tk.Tk()
        app = ReplayViewerGUI(root)
        root.mainloop()

if __name__ == "__main__":
    main()
