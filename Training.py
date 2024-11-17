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

    # Task 1: Data Cleaning - Handling missing values
    def handle_missing_values(self, dataframe):
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

    # Task 3: Normalization and standardization
    def standardization(self, dataframe, col_name):
        dataframe[col_name] = self.scaler.fit_transform(dataframe[[col_name]])
        print(f"Standardization applied to {col_name}.")

    def normalize_data(self, dataframe, col_name):
        dataframe[col_name] = self.normalizer.fit_transform(dataframe[[col_name]])
        print(f"Normalization applied to {col_name}.")

    # Task 1: Data Cleaning - Handling outliers
    def remove_outlier(self, data_frame, column_name, isTest=False):
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


    
    # Task 2: Feature Engineering - Encoding categorical variables
    def one_hot_encode(self, dataframe, column):
        print(f"Applying one-hot encoding to {column}...")
        encoded_df = pd.DataFrame(self.one_hot_encoder.fit_transform(dataframe[[column]]))
        encoded_df.columns = self.one_hot_encoder.get_feature_names_out([column])
        dataframe = dataframe.drop(column, axis=1).join(encoded_df)
        print(f"One-hot encoding applied to {column}.")
        return dataframe

    # Task 2: Feature Engineering - Feature selection
    # def feature_selection(self, dataframe):
    #     print("Performing feature selection...")

    #     # Filter out non-numeric columns
    #     numeric_df = dataframe.select_dtypes(include=['float64', 'int64']).copy()

    #     # Drop columns with any NaN values or fill them with 0, depending on your preference
    #     numeric_df = numeric_df.dropna(axis=1)  # or use numeric_df.fillna(0)

    #     # Variance threshold
    #     var_thres = VarianceThreshold(threshold=0)
    #     var_thres.fit(numeric_df)
    #     dropable_const_cols = numeric_df.columns[[not col for col in var_thres.get_support()]]
    #     numeric_df = numeric_df.drop(dropable_const_cols, axis=1)
    #     print(f"Low variance features removed: {list(dropable_const_cols)}")

    #     # Correlation matrix and removal of highly correlated features
    #     correlation_matrix = np.corrcoef(numeric_df, rowvar=False)
    #     correlated_features = [numeric_df.columns[x[0]] for x in self.get_correlated_features(correlation_matrix, 0.95)]
    #     numeric_df = numeric_df.drop(correlated_features, axis=1)
    #     print(f"Highly correlated features removed: {correlated_features}")

    #     return numeric_df

    # def get_correlated_features(self, correlation_matrix, threshold=0.8):
    #     num_features = correlation_matrix.shape[0]
    #     correlated_features = set()
    #     for i in range(num_features):
    #         for j in range(i + 1, num_features):
    #             correlation = correlation_matrix[i, j]
    #             if correlation >= threshold:
    #                 correlated_features.add((i, j))
    #     return correlated_features

    # Task 4: Handling class imbalance
    def hybrid_sampling(self, X, y):
        print("Applying hybrid sampling using SMOTEENN...")
        smote_enn = SMOTEENN()
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        print("Hybrid sampling applied successfully.")
        return X_resampled, y_resampled

    def preprocess_data(self, dataframe, isTest=False):
        print(f"Preprocessing {'test' if isTest else 'training'} data...")
        self.handle_missing_values(dataframe)
        for col in ['dur', 'spkts', 'sloss', 'dloss']:
            self.remove_outlier(dataframe, col, isTest)

        for col in ['proto', 'service', 'state']:
            dataframe = self.one_hot_encode(dataframe, col)

        numerical_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            self.standardization(dataframe, col)
            self.normalize_data(dataframe, col)

        dataframe = dataframe.drop(columns=['swin', 'stcpb', 'dtcpb', 'dwin', 'attack_cat'], axis=1, errors='ignore')
        dataframe = dataframe.replace({True: 0, False: 1})
        print("Preprocessing completed.")
        return dataframe


    def train_test_split(self):
        print("Splitting data into training, validation, and test sets...")
        
        # Load data from CSV files in chunks and preprocess
        self.train_df = self.load_data(self.train_file_path)
        self.test_df = self.load_data(self.test_file_path)
        
        # Preprocess data
        self.train_df = self.preprocess_data(self.train_df)
        self.test_df = self.preprocess_data(self.test_df, isTest=True)

        # Split the processed training data into train and temp (for validation and test)
        X_temp = self.train_df.drop('label', axis=1)

        y_temp = self.train_df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )

        # Check if y_train has more than one class before applying SMOTEENN
        # if len(y_train.unique()) > 1:
        #     X_train, y_train = self.hybrid_sampling(X_train, y_train)
        # else:
        #     print("Warning: y_train has only one class. Skipping hybrid sampling.")

        # # Feature selection on X_train only
        # X_train = self.feature_selection(X_train)

        # Ensure X_val and X_test have the same columns as X_train after feature selection
        X_test = X_test[X_train.columns]

        print("\nShapes before PCA:")
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)

        # Apply PCA on all sets
        n_components = min(15, X_train.shape[1])  # Set n_components <= minimum feature count
        pca = PCA(n_components=n_components)

        # Fit PCA on X_train and transform both X_train and X_test
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        print("Data splitting and processing completed.")
        # return X_train, X_val, X_test, y_train, y_val, y_test
        return X_train, X_test, y_train, y_test


# Paths to the data files
train_file_path = 'UNSW_NB15_training-set.csv'
test_file_path = 'UNSW_NB15_testing-set.csv'

# Instantiate the Preprocess class and run the preprocessing
processed = Preprocess(train_file_path, test_file_path, chunk_size=1000)
X_train, X_test, y_train, y_test = processed.train_test_split()


# SAVING DATA
# Convert numpy arrays back to DataFrames with dummy column names
X_train_df = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(X_train.shape[1])])
# X_val_df = pd.DataFrame(X_val, columns=[f'PC{i+1}' for i in range(X_val.shape[1])])
X_test_df = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(X_test.shape[1])])

# Convert target variables to DataFrames if they are not already
y_train_df = pd.DataFrame(y_train, columns=['label'])
# y_val_df = pd.DataFrame(y_val, columns=['label'])
y_test_df = pd.DataFrame(y_test, columns=['label'])

# Save to CSV
X_train_df.to_csv('processed_X_train.csv', index=False)
# X_val_df.to_csv('processed_X_val.csv', index=False)
X_test_df.to_csv('processed_X_test.csv', index=False)
y_train_df.to_csv('processed_y_train.csv', index=False)
# y_val_df.to_csv('processed_y_val.csv', index=False)
y_test_df.to_csv('processed_y_test.csv', index=False)

print("Processed data saved to CSV files.")

