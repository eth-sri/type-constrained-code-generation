import os
import sys
import glob


def convert_files(directory):
    # Define the pattern to match files
    pattern = os.path.join(directory, "*_nc.jsonl")

    # Find all files matching the pattern
    files = glob.glob(pattern)

    for file_path in files:
        # Extract the base name of the file
        base_name = os.path.basename(file_path)

        # Check if the file does not end with _repair-all or _translate before _c.jsonl or _nc.jsonl
        if not any(s in file_path for s in ("translate", "repair")):
            # Find the position of the last underscore before _c.jsonl or _nc.jsonl
            underscore_pos = base_name.rfind("_nc.jsonl")

            if underscore_pos != -1:
                # Extract the part before the underscore
                prefix = base_name[:underscore_pos]

                # Create the new file name
                new_file_name = f"{prefix}_synth_nc.jsonl"

                # Define the new file path
                new_file_path = os.path.join(directory, new_file_name)

                # Rename the file
                os.rename(file_path, new_file_path)
                print(f"Renamed '{file_path}' to '{new_file_path}'")


# Specify the directory containing the files
directory = sys.argv[1]

# Call the function to convert files
convert_files(directory)
