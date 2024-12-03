import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, optimizers
from sklearn.preprocessing import MinMaxScaler
import joblib
# import seaborn as sns
# import matplotlib.pyplot as plt

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow messages

# Configuration
BATCH_SIZE = 1000
EPOCHS = 5
LATENT_DIM = 100
LEARNING_RATE = 0.001
BETA_1 = 0.5


if (choice := input('t for train, other for test\n').lower().rstrip()) == 't':
    print('Training selected')
    # Paths
    INPUT_PATH = "processed_train.csv"
# else:
#     print('Testing selected')
#     INPUT_PATH = "processed_test.csv"

# Load and preprocess data
data = pd.read_csv(INPUT_PATH)

# Separate data into "Normal" and "Not Normal"
not_normal_data = data[data['label'] == 1]
normal_data = data[data['label'] == 0]

# Select columns for GAN and drop unnecessary columns
data_for_gan = not_normal_data.drop(columns=['proto', 'state', 'attack_cat', 'label'], errors='ignore')

# Replace '-' with 0 and ensure all data is numeric
data_for_gan.replace('-', 0, inplace=True)
data_for_gan = data_for_gan.apply(pd.to_numeric, errors='coerce').fillna(0)

# Normalize data
scaler = MinMaxScaler(feature_range=(-1, 1))
data_for_gan = scaler.fit_transform(data_for_gan)

# Generator model
def build_generator():
    model = models.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(data_for_gan.shape[1], activation='tanh')
    ])
    return model

# Discriminator model
def build_discriminator():
    model = models.Sequential([
        layers.Input(shape=(data_for_gan.shape[1],)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Initialize models
generator = build_generator()
discriminator = build_discriminator()

# Compile discriminator
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1),
    metrics=['accuracy']
)

# Combined GAN model
discriminator.trainable = False
gan_input = layers.Input(shape=(LATENT_DIM,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = models.Model(gan_input, gan_output)

gan.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
)

# Training loop
real_labels = np.ones((BATCH_SIZE, 1))
fake_labels = np.zeros((BATCH_SIZE, 1))

progress_intervals = EPOCHS // 10

for epoch in range(EPOCHS):
    try:
        # Train discriminator
        for start in range(0, data_for_gan.shape[0], BATCH_SIZE):
            # Generate indices for the current batch
            idx = np.arange(start, min(start + BATCH_SIZE, data_for_gan.shape[0]))
            
            # Get the real samples for the current batch using the indices
            real_samples = data_for_gan[idx]

            # Train discriminator
            noise = np.random.normal(0, 1, (len(real_samples), LATENT_DIM))  # Noise matches batch size
            fake_samples = generator.predict(noise, verbose=0)

            real_labels = np.ones((len(real_samples), 1))
            fake_labels = np.zeros((len(real_samples), 1))

            d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            noise = np.random.normal(0, 1, (len(real_samples), LATENT_DIM))
            g_loss = gan.train_on_batch(noise, real_labels)


            # Print progress at intervals
            print(f"batch: {start}")

    except Exception as e:
        print(f"Error during training at epoch {epoch}: {e}")
        break

# Generate synthetic samples
num_synthetic_samples = len(normal_data) - len(not_normal_data)
noise = np.random.normal(0, 1, (num_synthetic_samples, LATENT_DIM))
synthetic_data = generator.predict(noise, verbose=0)
synthetic_data = scaler.inverse_transform(synthetic_data)

# Create column names for synthetic data
synthetic_columns = not_normal_data.drop(columns=['proto', 'state', 'attack_cat', 'label'], errors='ignore').columns

# Combine synthetic data with real data
synthetic_df = pd.DataFrame(synthetic_data, columns=synthetic_columns)
synthetic_df['label'] = 1
combined_data = pd.concat([data, synthetic_df], ignore_index=True)
joblib.dump(generator, 'generator_model.pkl')
joblib.dump(discriminator, 'discriminator_model.pkl')

# Split into features (X_train) and labels (y_train)
X_train = combined_data.drop(columns=['label'])
y_train = combined_data['label']

if choice == 't':
    # Save the features and labels as separate CSV files
    X_train.to_csv('processed_X_train.csv', index=False)
    y_train.to_csv('processed_y_train.csv', index=False)
    print(f"Generated and combined data saved to processed_X_train and processed_y_train")
# else:
#     INPUT_PATH = "processed_test.csv"
#     X_train.to_csv('processed_X_test.csv', index=False)
#     y_train.to_csv('processed_y_test.csv', index=False)
#     print(f"Generated and combined data saved to processed_X_test and processed_y_test")

print("\nStatistics of Real Data:")
print(not_normal_data.describe())

print("\nStatistics of Synthetic Data:")
print(synthetic_df.describe())


# output_dir = "kde_plots"
# os.makedirs(output_dir, exist_ok=True)

# # Loop through each column and save the plots
# for column in not_normal_data.columns:
#     plt.figure(figsize=(8, 4))
#     sns.kdeplot(not_normal_data[column], label="Real", shade=True)
#     sns.kdeplot(synthetic_df[column], label="Synthetic", shade=True)
#     plt.title(f"Distribution of {column}")
#     plt.legend()

#     # Save the figure to the output directory
#     output_path = os.path.join(output_dir, f"{column}_distribution.png")
#     plt.savefig(output_path, format="png", dpi=300)

#     # Close the figure to free up memory
#     plt.close()
