import uuid
import os

def get_or_create_unique_id(config_path="config.txt", ids_file="IDs.txt"):
    """
    Reads a random ID from a config file.
    If not found, generates a new UUID4 that is unique across IDs listed in IDs.txt,
    writes it to both the config file and appends it to IDs.txt.
    
    Parameters:
    -----------
    config_path : str
        Path to the local config file
    ids_file : str
        Path to the file containing all used IDs
    
    Returns:
    --------
    str
        A unique random ID
    """
    random_id = None

    # 1. Check if the config file exists and contains an ID
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            for line in f:
                if line.startswith("id="):
                    random_id = line.strip().split("=", 1)[1]
                    break

    # 2. Read all used IDs
    used_ids = set()
    if os.path.exists(ids_file):
        with open(ids_file, "r") as f:
            used_ids = set(line.strip() for line in f if line.strip())

    # 3. If ID not found or ID is already in used_ids, generate a unique one
    if not random_id or random_id in used_ids:
        while True:
            random_id = str(uuid.uuid4())
            if random_id not in used_ids:
                break
        # Append the new ID to IDs.txt
        with open(ids_file, "a") as f:
            f.write(f"{random_id}\n")
        # Write the new ID to the config file
        with open(config_path, "w") as f:
            f.write(f"id={random_id}\n")

    return random_id
