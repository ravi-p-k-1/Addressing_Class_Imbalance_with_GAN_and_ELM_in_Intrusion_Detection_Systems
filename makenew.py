import pandas as pd
from sklearn.model_selection import train_test_split

# File paths
input_file = "UNSW-NB15_2.csv"
training_file = "new_training.csv"
testing_file = "new_testing.csv"

# Total lines in the file
total_lines = 672818

# Calculate 15% for training and 5% for testing
training_size = int(total_lines * 0.15)
testing_size = int(total_lines * 0.05)

# Original headers from your list
original_headers = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl",
    "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin",
    "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "attack_cat", "label"
]

# Remove columns 1-4, 29-30 from headers (0-based indices: 0-3, 28-29)
headers_to_keep = [col for i, col in enumerate(original_headers) if i not in [0, 1, 2, 3, 28, 29]]

# Read the file in chunks to avoid memory issues
chunk_size = 10000
chunks = []
reader = pd.read_csv(input_file, chunksize=chunk_size)

# Collect and concatenate chunks
for chunk in reader:
    chunks.append(chunk)

data = pd.concat(chunks, ignore_index=True)

# Keep only the desired columns based on index removal
columns_to_keep_indices = [i for i in range(len(original_headers)) if i not in [0, 1, 2, 3, 28, 29]]
data_cleaned = data.iloc[:, columns_to_keep_indices]

# Update headers
data_cleaned.columns = headers_to_keep

data_cleaned_copy=data_cleaned.copy()

# Create a column for stratification (based on the combination of categorical features)
data_cleaned["stratify_col"] = data_cleaned_copy["proto"].astype(str) + "_" + data_cleaned_copy["state"].astype(str) + "_" + data_cleaned_copy["service"].astype(str)

# Identify rare classes (combinations with only 1 occurrence)
combination_counts = data_cleaned["stratify_col"].value_counts()
rare_combinations = combination_counts[combination_counts == 1].index

# # Shuffle data
# data_shuffled = data_cleaned.sample(frac=1, random_state=42).reset_index(drop=True)

# # Split into training and testing sets
# training_data = data_shuffled[:training_size]
# testing_data = data_shuffled[training_size:training_size + testing_size]

# # Save training and testing sets to separate files
# training_data.to_csv(training_file, index=False)
# testing_data.to_csv(testing_file, index=False)

# print(f"15% of the lines with selected columns removed have been saved into {training_file}.")
# print(f"5% of the lines with selected columns removed have been saved into {testing_file}.")

# Split the dataset
train_data, temp_data = train_test_split(
    data_cleaned[~data_cleaned["stratify_col"].isin(rare_combinations)], 
    test_size=(0.15), 
    stratify=data_cleaned[~data_cleaned["stratify_col"].isin(rare_combinations)]["stratify_col"], 
    random_state=42
)

# Extract the rows with rare combinations
rare_train_data = data_cleaned[data_cleaned["stratify_col"].isin(rare_combinations)]
rare_test_data = rare_train_data.copy()  # Add all rare combinations to test data as well

# Combine the datasets back together
train_data = pd.concat([train_data, rare_train_data])
test_data = pd.concat([temp_data, rare_test_data])

train_data = train_data.drop(columns=["stratify_col"])
test_data = test_data.drop(columns=["stratify_col"])


# Save the training and testing sets to separate files
train_data.to_csv(training_file, index=False)
test_data.to_csv(testing_file, index=False)

print(f"15% of the lines with selected columns removed have been saved into {training_file}.")
print(f"5% of the lines with selected columns removed have been saved into {testing_file}.")
