# New database handler
from pathlib import Path
import os
from database_handler import SimulationDB
base_dir = Path(__file__).resolve().parent.parent
results_dir = base_dir / "results"
# --- Database Integration Step ---
print("\n--- Starting Database Integration ---")
# Have to do this hacky thing to make sure the db isn't accessed concurrently and breaks
# Define the master database
master_db_path = results_dir / 'simulation_results_v2.json'
master_db = SimulationDB(db_path=master_db_path)

# Find all individual database files
individual_db_files = list(results_dir.glob('simulation_results_v2_*.json'))

if not individual_db_files:
    print("No individual database files found to merge.")
else:
    print(f"Found {len(individual_db_files)} individual database files to merge into {master_db_path.name}")
    
    for file_path in individual_db_files:
        try:
            temp_db = SimulationDB(db_path=file_path)
            records = temp_db.all()
            if records:
                # Insert records one by one to avoid document ID conflicts.
                # The upsert logic in master_db.insert() will handle duplicates based on parameters.
                for record in records:
                    master_db._db.insert(dict(record))
            temp_db.close()
            os.remove(file_path) # Clean up the individual file
        except Exception as e:
            print(f"Could not process or delete {file_path.name}. Error: {e}")

master_db.close()
print("--- Database Integration Finished ---")