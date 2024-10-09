import os
import glob
import re
import pandas as pd

# Directory containing the CSV files
directory_path = "/Users/yuenc2/Desktop/gender-bias/data/outputs/s0_1007/"

# Get all CSV files in the directory
csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

# Loop through each CSV file and apply the operations
for csv_file in csv_files:
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_file)

    # Print column names for inspection (debugging purposes)
    print(f"Columns in {csv_file}: {df.columns.tolist()}")

    # Identify prefixes for columns (assuming consistent pattern)
    # Get a sample column that matches the pattern for finding the prefix (for 'implicit' and 'explicit')
    sample_col_implicit = next((col for col in df.columns if re.search(r'_implicit\d+$', col)), None)
    sample_col_explicit = next((col for col in df.columns if re.search(r'_explicit\d+$', col)), None)

    for sample_col in [sample_col_implicit, sample_col_explicit]:
        if sample_col:
            # Extract the prefix dynamically using regex (everything before '_male_implicit' or '_female_explicit')
            match = re.match(r'(.*)_(male|female|diverse)_(implicit|explicit)\d+$', sample_col)
            if match:
                prefix = match.group(1)
                category = match.group(3)  # Either 'implicit' or 'explicit'

                # Drop the old columns without the ".1" suffix
                for prompt_id in range(25):  # Assuming 25 prompts, adjust as needed
                    columns_to_drop = [
                        f'{prefix}_male_{category}{prompt_id}',
                        f'{prefix}_female_{category}{prompt_id}',
                        f'{prefix}_diverse_{category}{prompt_id}'
                    ]
                    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

                # Rename columns with ".1" suffix back to original names
                for prompt_id in range(25):
                    columns_to_rename = {
                        f'{prefix}_male_{category}{prompt_id}.1': f'{prefix}_male_{category}{prompt_id}',
                        f'{prefix}_female_{category}{prompt_id}.1': f'{prefix}_female_{category}{prompt_id}',
                        f'{prefix}_diverse_{category}{prompt_id}.1': f'{prefix}_diverse_{category}{prompt_id}'
                    }
                    df.rename(columns={k: v for k, v in columns_to_rename.items() if k in df.columns}, inplace=True)

    # Save the updated DataFrame back to the CSV file (overwriting the original)
    df.to_csv(csv_file, index=False)

    print(f"Processed and updated file: {csv_file}")
