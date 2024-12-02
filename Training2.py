import csv
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN

class Preprocess:
    def __init__(self, train_file_path, test_file_path, chunk_size=1000, max_rows=10000000000):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.chunk_size = chunk_size
        self.max_rows = max_rows  # Maximum rows to load for processing
        self.lb_ub = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.normalizer = MinMaxScaler()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        print("Initialization complete. Ready to process data in chunks.")

    def read_csv_limited(self, file_path):
        """Generator to read the CSV file in limited chunks, up to max_rows."""
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            yield header
            
            chunk = []
            total_rows = 0
            for i, row in enumerate(reader):
                if total_rows >= self.max_rows:
                    break
                if i % self.chunk_size == 0 and i > 0:
                    yield pd.DataFrame(chunk, columns=header)
                    chunk = []
                chunk.append(row)
                total_rows += 1
            if chunk:
                yield pd.DataFrame(chunk, columns=header)

    def load_data(self, file_path):
        """Load a limited number of rows from the CSV in chunks and concatenate."""
        print(f"Loading up to {self.max_rows} rows from {file_path} in chunks...")
        chunks = []
        total_rows_loaded = 0
        for chunk in self.read_csv_limited(file_path):
            if isinstance(chunk, pd.DataFrame):  # Skip header row
                chunks.append(chunk)
                total_rows_loaded += len(chunk)
                if total_rows_loaded >= self.max_rows:
                    break
        data = pd.concat(chunks, ignore_index=True).iloc[:self.max_rows]
        print(f"Data loaded from {file_path} with {data.shape[0]} rows.")
        return data

    def handle_missing_values(self, dataframe):
        """Handle missing values in numerical and categorical columns."""
        print("Handling missing values...")
        numerical_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            dataframe[numerical_cols] = imputer.fit_transform(dataframe[numerical_cols])

        categorical_cols = dataframe.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            dataframe[categorical_cols] = imputer.fit_transform(dataframe[categorical_cols])

        print("Missing values handled successfully.")

    def remove_outlier(self, data_frame, column_name, isTest=False):
        """Remove outliers using IQR for training data and use stored bounds for test data."""
        if column_name not in data_frame.select_dtypes(include=['float64', 'int64']).columns:
            print(f"Skipping outlier removal for non-numeric column: {column_name}")
            return

        if isTest:
            data_frame.loc[data_frame[column_name] > self.lb_ub[column_name][1], column_name] = self.lb_ub[column_name][1]
            data_frame.loc[data_frame[column_name] < self.lb_ub[column_name][0], column_name] = self.lb_ub[column_name][0]
            print(f"Outliers handled for {column_name} in test data.")
            return

        q1 = data_frame[column_name].quantile(0.25)
        q3 = data_frame[column_name].quantile(0.75)
        iqr = q3 - q1
        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr

        data_frame.loc[data_frame[column_name] > ub, column_name] = ub
        data_frame.loc[data_frame[column_name] < lb, column_name] = lb
        self.lb_ub[column_name] = (lb, ub)
        print(f"Outliers removed for {column_name}. Lower bound: {lb}, Upper bound: {ub}")

    def encode_categorical(self, dataframe, column):
        """Apply label encoding to a categorical column."""
        print(f"Applying label encoding to {column}...")
        dataframe[column] = self.label_encoder.fit_transform(dataframe[column].astype(str))
        print(f"Label encoding applied to {column}.")
        return dataframe

    def hybrid_sampling(self, X, y):
        """Apply hybrid sampling using SMOTEENN to handle class imbalance."""
        print("Applying hybrid sampling using SMOTEENN...")
        smote_enn = SMOTEENN()
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        print("Hybrid sampling applied successfully.")
        return X_resampled, y_resampled

    def preprocess_data(self, dataframe, isTest=False):
        """Preprocess the data, handling missing values, outliers, and encoding."""
        print(f"Preprocessing {'test' if isTest else 'training'} data...")
        self.handle_missing_values(dataframe)

        # Handle outliers for specific columns
        for col in ['dur', 'spkts', 'sloss', 'dloss']:
            self.remove_outlier(dataframe, col, isTest)

        # label encode categorical columns
        for col in ['proto', 'service', 'state']:
            dataframe = self.encode_categorical(dataframe, col)

        # Apply standardization and normalization to numerical columns
        numerical_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            self.standardization(dataframe, col)
            self.normalize_data(dataframe, col)

        dataframe = dataframe.drop(columns=['swin', 'stcpb', 'dtcpb', 'dwin', 'attack_cat'], axis=1, errors='ignore')
        dataframe = dataframe.replace({True: 0, False: 1})
        print("Preprocessing completed.")
        return dataframe

    def standardization(self, dataframe, col_name):
        """Standardize a column of data."""
        dataframe[col_name] = self.scaler.fit_transform(dataframe[[col_name]])
        print(f"Standardization applied to {col_name}.")

    def normalize_data(self, dataframe, col_name):
        """Normalize a column of data."""
        dataframe[col_name] = self.normalizer.fit_transform(dataframe[[col_name]])
        print(f"Normalization applied to {col_name}.")

    def train_test_split(self):
        """Load data, preprocess, and split into training and test sets."""
        print("Splitting data into training and test sets...")

        # Load data
        train_df = self.load_data(self.train_file_path)
        test_df = self.load_data(self.test_file_path)

        # Preprocess data
        train_df = self.preprocess_data(train_df)
        test_df = self.preprocess_data(test_df, isTest=True)

        # Separate features and labels
        X_train = train_df.drop('label', axis=1)
        y_train = train_df['label']
        X_test = test_df.drop('label', axis=1)
        y_test = test_df['label']
        
        # Ensure numeric values and handle NaN
        X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Apply hybrid sampling if needed (example)
        # if len(y_train.unique()) > 1:
        #     X_train, y_train = self.hybrid_sampling(X_train, y_train)

        # Print shapes before PCA
        print("\nShapes before PCA:")
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)

        # Apply PCA
        n_components = min(15, X_train.shape[1])  # Set n_components <= minimum feature count
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        print("PCA applied successfully.")
        return X_train, X_test, y_train, y_test


# Paths to the data files
# train_file_path = 'new_GAN_training.csv'
train_file_path = 'new_training.csv'
test_file_path = 'new_testing.csv'

# Instantiate the Preprocess class and run the preprocessing
processed = Preprocess(train_file_path, test_file_path, chunk_size=1000)
X_train, X_test, y_train, y_test = processed.train_test_split()

# Save processed data to CSV

train_data = pd.concat([
    pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(X_train.shape[1])]),
    pd.DataFrame(y_train, columns=['label'])
], axis=1)

# Combine X_test and y_test into a single DataFrame
test_data = pd.concat([
    pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(X_test.shape[1])]),
    pd.DataFrame(y_test, columns=['label'])
], axis=1)

# Save combined DataFrames to CSV
train_data.to_csv('processed_train.csv', index=False)
test_data.to_csv('processed_test.csv', index=False)

# pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(X_train.shape[1])]).to_csv('processed_X_train.csv', index=False)
pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(X_test.shape[1])]).to_csv('real_processed_X_test.csv', index=False)
# pd.DataFrame(y_train, columns=['label']).to_csv('processed_y_train.csv', index=False)
pd.DataFrame(y_test, columns=['label']).to_csv('real_processed_y_test.csv', index=False)

print("Processed data saved to CSV files.")
