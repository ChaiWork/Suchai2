import os
import tarfile

def main():
    # Mapping of local file path to path in the root of the archive
    files_to_include = [
        ("main.py", "main.py"),
        ("deck.csv", "deck.csv"),
        ("src/model.py", "model.py"),
        ("src/agent.py", "agent.py"),
        ("cg", "cg")
    ]
    
    # Prioritize best_model.pth (highest evaluation win rate) over final model.pth
    if os.path.exists("best_model.pth"):
        files_to_include.append(("best_model.pth", "model.pth"))
        print("Packaging best_model.pth in submission archive...")
    elif os.path.exists("model.pth"):
        files_to_include.append(("model.pth", "model.pth"))
        print("Packaging root model.pth in submission archive...")
    elif os.path.exists("out/model5.pth"):
        files_to_include.append(("out/model5.pth", "model.pth"))
        print("Packaging out/model5.pth as model.pth in submission archive...")
    else:
        print("Warning: No model weights found. The submission will not include trained model weights.")

    output_filename = "submission.tar.gz"
    
    print(f"Creating {output_filename}...")
    with tarfile.open(output_filename, "w:gz") as tar:
        for local_path, arc_name in files_to_include:
            if os.path.exists(local_path):
                print(f"  Adding {local_path} -> {arc_name}")
                # Add to tar archive at the root directory level (not nested)
                tar.add(local_path, arcname=arc_name)
            else:
                print(f"  Error: Required file '{local_path}' does not exist!")
                return
                
    # Verify the final archive size against constraints (197.7 MiB)
    size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print(f"Created {output_filename} successfully.")
    print(f"Size: {size_mb:.2f} MiB (Limit: 197.7 MiB)")

if __name__ == "__main__":
    main()
