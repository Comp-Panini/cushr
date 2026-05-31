import os
import random
import shutil
import glob

def sample_graphml_files(num_files=50):
    source_dir = "../../SIGHUM_database/After_graphml"
    dest_dir = "./test_batch"

    # verify directories exist
    if not os.path.exists(source_dir):
        print(f"error: could not find source directory at '{source_dir}'")
        return

    # create test_batch folder
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"created destination directory: {dest_dir}")

    # find all .graphml files in the source folder
    print(f"scanning '{source_dir}' for files...")
    all_files = glob.glob(os.path.join(source_dir, "*.graphml"))
    
    total_files = len(all_files)
    if total_files == 0:
        print("error: no .graphml files found in the source directory")
        return
        
    print(f"found {total_files} files in total")

    # pick 50 random files & copy over
    sample_size = min(num_files, total_files)
    selected_files = random.sample(all_files, sample_size)
    print(f"copying {sample_size} random files to '{dest_dir}'...")
    
    copied_count = 0
    for file_path in selected_files:
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(dest_dir, file_name)
        
        try:
            shutil.copy2(file_path, destination_path)
            copied_count += 1
        except Exception as e:
            print(f"failed to copy {file_name}: {e}")

    print(f"\nsuccess! copied {copied_count} files into test_batch folder")

if __name__ == "__main__":
    sample_graphml_files(50)