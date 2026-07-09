import os
import tarfile

def main():
    # List of files to package in the root of the submission bundle
    files_to_include = [
        "main.py",
        "deck.csv",
        "model.py",
        "agent.py"
    ]
    
    if os.path.exists("model.pth"):
        files_to_include.append("model.pth")
    else:
        print("Warning: model.pth not found. The submission will not include trained model weights.")

    output_filename = "submission.tar.gz"
    
    print(f"Creating {output_filename}...")
    with tarfile.open(output_filename, "w:gz") as tar:
        for file in files_to_include:
            if os.path.exists(file):
                print(f"  Adding {file}")
                # Add to tar archive at the root directory level (not nested)
                tar.add(file, arcname=os.path.basename(file))
            else:
                print(f"  Error: Required file '{file}' does not exist!")
                return
                
    # Verify the final archive size against constraints (197.7 MiB)
    size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print(f"Created {output_filename} successfully.")
    print(f"Size: {size_mb:.2f} MiB (Limit: 197.7 MiB)")

if __name__ == "__main__":
    main()
