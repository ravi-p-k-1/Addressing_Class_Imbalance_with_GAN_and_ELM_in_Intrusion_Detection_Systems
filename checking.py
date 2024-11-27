import pandas as pd

# Load your train data (replace with your file path)
train_data = pd.read_csv('new_training.csv')  # Adjust to your actual file name

original_headers = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl",
    "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin",
    "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "attack_cat", "label"
]

headers_to_keep = [col for i, col in enumerate(original_headers) if i not in [0, 1, 2, 3, 28, 29]]

# train_data.columns = original_headers
train_data.columns = headers_to_keep

# Extract the 'label' column as y_train
y_train = train_data['label']

# Check for NaN values in y_train
print("Number of NaN values in y_train:", y_train.isnull().sum())
