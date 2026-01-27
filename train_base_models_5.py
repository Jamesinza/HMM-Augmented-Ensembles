import gc
import math
import random
import scipy
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, saving
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from hmmlearn.hmm import GaussianHMM, CategoricalHMM, GMMHMM, PoissonHMM, MultinomialHMM

seed=42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def load_base_models(sub_folder, datasets, optimizers, dims, seeds):
    for dataset in datasets:
        for optimizer in optimizers:
            for dim in dims:
                for seed in seeds:
                    model = saving.load_model(f'test_models/{sub_folder}/{dataset}_{optimizer}_dim{dim}_seed{seed}.keras')
                    yield model
                    del model    

def compute_batch_size(dataset_length):
    base_unit  = 25000
    base_batch = 32
    batch_size = base_batch * math.ceil(dataset_length / base_unit)
    return batch_size    

# --- Evaluation ---
def evaluate_model(model, X_test, name):
    pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, pred)
    print(f"{name} Accuracy: {acc:.4f}")
    return acc

# --- Base Model Generator ---
def create_base_model(dataset, optimizer, dim, seed, dropout, num_classes):
    inputs = layers.Input(input_shape)
    x1 = x2 = x3 = inputs

    for _ in range(2):
        x1 = layers.GRU(dim, return_sequences=True, seed=seed)(x1)
        x1 = layers.Dropout(dropout, seed=seed)(x1)
    for _ in range(2):
        x2 = layers.LSTM(dim, return_sequences=True, seed=seed)(x2)
        x2 = layers.Dropout(dropout, seed=seed)(x2)
    x3 = layers.Add()([x1, x2])
        
    out = layers.GlobalAveragePooling1D()(x3)
    out = layers.Dense(dim // 2, activation='relu')(out)
    out = layers.Dense(num_classes, activation='softmax')(out)

    model = models.Model(inputs, out, name=f'{dataset}_{optimizer}_{dim}_{seed}')
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return model

# --- Train Base Models ---
def train_base_models(dataset, optimizers, dims, seeds, dropout, num_classes, batch_size,
                      epochs, sub_folder, X_train, X_val, y_train, y_val):
    for optimizer in optimizers:
        for dim in dims:
            for seed in seeds:
                tf.keras.backend.clear_session()
                gc.collect()                 
                callback = [
                    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
                ]
                model = create_base_model(dataset, optimizer, dim, seed, dropout, num_classes)
                print(f'\nTraining {dataset}_{optimizer}_dim{dim}_seed{seed}...')
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                          verbose=1, callbacks=callback, class_weight=class_weights_dict)
                model.save(f'test_models/{sub_folder}/{dataset}_{optimizer}_dim{dim}_seed{seed}.keras')
                del model

# --- Get real world data ---
def get_real_data(dataset, num_samples=200_000):
    print(f'\nBuilding dataframe using {dataset} data...')
    df = pd.read_csv(f'datasets/training/{dataset}_Full.csv')

    col5 = ['Take5','Thunderball','Euro','Mega','Powerball']
    if dataset in col5:
        cols = ['A', 'B', 'C', 'D', 'E']
    else:
        cols = ['A', 'B', 'C', 'D', 'E', 'F']

    if dataset == 'Quick':
        df = df.drop(columns=['Unnamed: 0']).dropna().reset_index(drop=True).astype(np.int8)
    else:
        df = df[cols].dropna().reset_index(drop=True).astype(np.int8)        
    
    # Format each element as 2-digit string and then flatten digits into an array.
    df = df.map(lambda x: f'{x:02d}')
    flattened = df.values.flatten()
    full_data = np.array([int(d) for num in flattened for d in str(num)], dtype=np.int8)
    return full_data[-num_samples:]

# --- Generate data sequences ---
def generate_dataset(data, wl=10, features=1):
    X_test = np.empty([len(data)-wl, wl, features], dtype=np.float32)
    y_test = np.empty([len(data)-wl, 1], dtype=np.int8)
    for i in range(len(data)-wl):
        X_test[i] = data[i:i+wl]
        y_test[i] = data[i+wl, :1]    
    return X_test, y_test


# --- Parameters ---
num_classes = 10
epochs = 6
# batch_size = 512
sub_folder = 'M5'
wl = 10
dropout = 0.1
num_heads = 2
seeds = [42, 95, 138, 276, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,]
         # 233, 377, 610, 987, 1597, 2584, 4181, 6765,
         # 10946, 17711, 28657, 46368, 75025, 121393,
         # 196418, 317811, 514229]
dims = [128]
optimizers = ['adam','adamw','nadam','adamax']
# stats = [np.mean, np.std, np.ptp, scipy.stats.skew, scipy.stats.kurtosis]

datasets = ['Take5','Mega','Thunderball','Euro','Powerball','C4L','NYLot','HotPicks','Quick']
for dataset in datasets:
    X_raw = get_real_data(dataset, 200_000)
    data_raw = X_raw.reshape(-1,1)
    data = data_raw
    # data = np.load('data/Take5_stats_features.npy')

    batch_size = compute_batch_size(len(data))
    features = data.shape[-1]
    input_shape = (wl, features)
    print(f'\ninput_shape: {input_shape}\n')
    
    X, y = generate_dataset(data, wl, features)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    
    # Scaling data
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, X_train.shape[1]*X_train.shape[2])
    X_val_2d = X_val.reshape(-1, X_val.shape[1]*X_val.shape[2])
    X_test_2d = X_test.reshape(-1, X_test.shape[1]*X_test.shape[2])
    
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_val_scaled = scaler.transform(X_val_2d)
    X_test_scaled = scaler.transform(X_test_2d)

    joblib.dump(scaler, f'test_models/{dataset}_base_scaler.joblib')
    
    del X_train_2d, X_val_2d, X_test_2d, scaler
    
    # Reshape back to 3D format
    X_train = X_train_scaled.reshape(-1, X_train.shape[1], X_train.shape[2])
    X_val = X_val_scaled.reshape(-1, X_val.shape[1], X_val.shape[2])
    X_test = X_test_scaled.reshape(-1, X_test.shape[1], X_test.shape[2])

    del X_train_scaled, X_val_scaled, X_test_scaled
    
    unique_classes = np.unique(X_raw.flatten())
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=X_raw.flatten())
    class_weights_dict = dict(enumerate(class_weights))
    
    train_base_models(dataset, optimizers, dims, seeds, dropout, num_classes, batch_size,
                                    epochs, sub_folder, X_train, X_val, y_train, y_val)

    del X_train, X_val, X_test
