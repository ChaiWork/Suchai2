import argparse
import os
import sys
import webbrowser
from pathlib import Path

# Add parent directory of this script to path to import viewer.py if run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import viewer

def find_default_pdf() -> Path | None:
    """Attempts to find the competition Card ID PDF in common paths."""
    search_dirs = [Path("."), Path(".."), Path("../..")]
    patterns = ["*Card_ID*.pdf", "Card_ID List_JP.pdf", "Card_ID List_EN.pdf"]
    
    # Check kaggle inputs structure
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        search_dirs.append(kaggle_input)
        
    for sd in search_dirs:
        if not sd.exists():
            continue
        # Check recursively in each search dir
        for pattern in patterns:
            try:
                matches = list(sd.rglob(pattern))
                if matches:
                    return matches[0]
            except Exception:
                pass
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Pokémon TCG Battle Challenge Replay Compiler & Viewer."
    )
    parser.add_argument(
        "json_path", 
        type=str, 
        help="Path to the battle replay JSON log file."
    )
    parser.add_argument(
        "--pdf", 
        type=str, 
        default=None, 
        help="Path to the competition Card_ID List PDF. If not specified, search will occur automatically."
    )
    parser.add_argument(
        "--img-dir", 
        type=str, 
        default="card_images", 
        help="Directory to cache extracted card PNG images (default: card_images)."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="battle_viewer.html", 
        help="Output HTML path (default: battle_viewer.html)."
    )
    parser.add_argument(
        "--no-images", 
        action="store_true", 
        help="Compile HTML without extracting or loading card images."
    )
    parser.add_argument(
        "--no-open", 
        action="store_true", 
        help="Do not automatically open the compiled viewer in the default browser."
    )
    
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: Replay JSON file '{json_path}' not found.")
        sys.exit(1)

    prepare_images = not args.no_images
    pdf_path = None

    if prepare_images:
        if args.pdf:
            pdf_path = Path(args.pdf)
        else:
            pdf_path = find_default_pdf()
            
        if pdf_path and pdf_path.exists():
            print(f"Found competition PDF at: {pdf_path}")
        else:
            if args.pdf:
                print(f"Error: Specified PDF '{args.pdf}' not found.")
                sys.exit(1)
            else:
                print("Warning: Could not find competition Card ID PDF automatically.")
                print("Replay will be compiled without extracting card image thumbnails.")
                print("Tip: Pass --pdf <path_to_pdf> to extract images.")
                prepare_images = False

    print("Compiling battle replay...")
    try:
        out_html = viewer.write_html(
            json_path=json_path,
            out_html=args.output,
            card_image_dir=args.img_dir,
            card_pdf=pdf_path,
            prepare_images=prepare_images,
            overwrite_images=False
        )
        print(f"Success! Compiled HTML viewer saved to: {out_html.resolve()}")
        
        if not args.no_open:
            file_url = f"file:///{str(out_html.resolve()).replace(os.sep, '/')}"
            print(f"Opening viewer in browser: {file_url}")
            webbrowser.open(file_url)
            
    except Exception as e:
        print(f"Error compiling replay: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
