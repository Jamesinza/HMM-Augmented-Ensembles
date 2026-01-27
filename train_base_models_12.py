# Silencing annoying warnings
import shutup
shutup.please()

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from hmmlearn.hmm import GaussianHMM, CategoricalHMM, GMMHMM, PoissonHMM, MultinomialHMM

seed=42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

@tf.keras.saving.register_keras_serializable()
class ForgetBiasInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        # Bias layout for LSTM: [input_gate, forget_gate, cell_gate, output_gate]
        result = tf.zeros(shape, dtype=dtype)
        n = shape[0] // 4
        result = tf.tensor_scatter_nd_update(result, [[n]], [1.0])
        return result

def hmm_normal_stack(X_raw, rngs):
    """
    Augment the raw input with HMM-based features.
    The raw data X_raw is assumed to be a 2D array.
    """
    base_features = None
    for rng in rngs:
        print(f'\nCreating HMM normal features using seed {rng}...')
        # Make copies for HMM-based features
        hs1 = X_raw.copy()
        hs2 = X_raw.copy()
        hs3 = X_raw.copy()
        hs4 = X_raw.copy()
        hs5 = X_raw.copy()

        # --- HMM Feature Augmentation ---
        # Loop over a single iteration or more if needed.
        a=b=c=d=e=10
        for _ in range(1):
            # GaussianHMM and CategoricalHMM
            # print('\nNow running GaussianHMM normal...')
            hmm_g = GaussianHMM(n_components=a, covariance_type="full", random_state=rng)
            # print('\nNow running CategoricalHMM normal...')
            hmm_c = CategoricalHMM(n_components=b, n_features=10, random_state=rng)
            hmm_g.fit(hs1)
            hmm_c.fit(hs2)
            hs1_pred = hmm_g.predict(hs1).reshape(-1, 1).astype(np.int8)
            a = len(np.unique(hs1_pred))
            # print(f'\na: {a}')
            hs2_pred = hmm_c.predict(hs2).reshape(-1, 1).astype(np.int8)
            b = len(np.unique(hs2_pred))
            # print(f'\nb: {b}')
            base_features = (np.hstack([base_features, hs1_pred, hs2_pred], dtype=np.int8)
                             if base_features is not None
                             else np.hstack([hs1_pred, hs2_pred], dtype=np.int8))
            hs1, hs2 = hs1_pred, hs2_pred  # Update for potential further iterations
            # print(f'\nhs1:\n{hs1}')
            # print(f'hs2:\n{hs2}\n')

        for i in range(1):
            # GMMHMM Feature
            # print('\nNow running GMMHMM stepped...')
            hmm_gmm = GMMHMM(n_components=c, n_mix=1, covariance_type="full", random_state=rng)
            hmm_gmm.fit(hs3)
            hs3_pred = hmm_gmm.predict(hs3).reshape(-1, 1).astype(np.int8)
            c = len(np.unique(hs3_pred))
            # print(f'\nc: {c}')
            base_features = (np.hstack([base_features, hs3_pred], dtype=np.int8)
                             if base_features is not None
                             else hs3_pred)
            hs3 = hs3_pred  # Update for potential further iterations
            # print(f'\nhs3:\n{hs3}')

        for _ in range(1):
            # PoissonHMM Feature
            # print('\nNow running PoissonHMM normal...')
            hmm_pois = PoissonHMM(n_components=d, random_state=rng)
            hmm_pois.fit(hs4)
            hs4_pred = hmm_pois.predict(hs4).reshape(-1, 1).astype(np.int8)
            d = len(np.unique(hs4_pred))
            # print(f'\nc: {c}')
            base_features = (np.hstack([base_features, hs4_pred], dtype=np.int8)
                             if base_features is not None
                             else hs4_pred)
            hs4 = hs4_pred  # Update for potential further iterations
            # print(f'\nhs4:\n{hs4}')

        for _ in range(1):
            # MultinomialHMM Feature
            # print('\nNow running MultinomialHMM normal...')
            hmm_multi = MultinomialHMM(n_components=e, random_state=rng)
            hmm_multi.fit(hs5)
            hs5_pred = hmm_multi.predict(hs5).reshape(-1, 1).astype(np.int8)
            # print(hs5_pred)
            e = len(np.unique(hs5_pred))
            # print(f'\ne: {e}')
            base_features = (np.hstack([base_features, hs5_pred], dtype=np.int8)
                             if base_features is not None
                             else hs5_pred)
            hs5 = hs5_pred  # Update for potential further iterations
            # print(f'\nhs5:\n{hs5}')
    return base_features


def hmm_stepped_stack(X_raw, rngs):
    """
    Augment the raw input with HMM-based features.
    In this version the n_components is reduced by 1
    with each iteration.
    The raw data X_raw is assumed to be a 2D array.
    """
    base_features = None
    for rng in rngs:
        print(f'\nCreating HMM stepped features using seed {rng}...')
        # Make copies for HMM-based features
        hs1 = X_raw.copy()
        hs2 = X_raw.copy()
        hs3 = X_raw.copy()

        # --- HMM Feature Augmentation ---
        # Loop over a single iteration or more if needed.
        a=b=9
        for i in range(1):
            # GaussianHMM and CategoricalHMM
            # print('\nNow running GaussianHMM stepped...')
            hmm_g = GaussianHMM(n_components=a - i, covariance_type="full", random_state=rng)
            # print('\nNow running CategoricalHMM stepped...')
            hmm_c = CategoricalHMM(n_components=b - i, n_features=10, random_state=rng)
            hmm_g.fit(hs1)
            hmm_c.fit(hs2)
            hs1_pred = hmm_g.predict(hs1).reshape(-1, 1).astype(np.int8)
            a = len(np.unique(hs1_pred))
            # print(f'\na: {a}')
            hs2_pred = hmm_c.predict(hs2).reshape(-1, 1).astype(np.int8)
            b = len(np.unique(hs2_pred))
            # print(f'\nb: {b}')
            base_features = (np.hstack([base_features, hs1_pred, hs2_pred], dtype=np.int8)
                             if base_features is not None
                             else np.hstack([hs1_pred, hs2_pred], dtype=np.int8))
            hs1, hs2 = hs1_pred, hs2_pred  # Update for potential further iterations
            # print(f'\nhs1:\n{hs1}')
            # print(f'hs2:\n{hs2}\n')
    return base_features


def get_extra_features(X_raw, rng):
    """
    Create HMM-based and NVG-based features.
    """
    hmm_features1 = hmm_normal_stack(X_raw, rng)
    # print(f'\nhmm_features1: {hmm_features1.shape}')
    hmm_features2 = hmm_stepped_stack(X_raw, rng)
    # print(f'hmm_features2: {hmm_features2.shape}')

    # Stack all new features.
    X_augmented = np.hstack([hmm_features1, hmm_features2], dtype=np.int8)
    # X_augmented = hmm_features1
    return X_augmented

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
def create_cnn_base_model(dataset, arch, optimizer, dim, seed, dropout, num_classes):
    k_init = tf.keras.initializers.HeUniform(seed=seed)
    inputs = tf.keras.layers.Input(input_shape)

    conv1=conv2=conv3=inputs
    for _ in range(2):
        conv1 = tf.keras.layers.Conv1D(filters=dim, kernel_size=1, padding="same", kernel_initializer=k_init)(conv1)
        conv1 = tf.keras.layers.Dropout(dropout, seed=seed)(conv1)
        conv1 = tf.keras.layers.LayerNormalization()(conv1)
        conv1 = tf.keras.layers.ReLU()(conv1)

    for _ in range(2):
        conv2 = tf.keras.layers.Conv1D(filters=dim, kernel_size=3, padding="same", kernel_initializer=k_init)(conv2)
        conv2 = tf.keras.layers.Dropout(dropout, seed=seed)(conv2)
        conv2 = tf.keras.layers.LayerNormalization()(conv2)
        conv2 = tf.keras.layers.ReLU()(conv2)

    for _ in range(2):
        conv3 = tf.keras.layers.Conv1D(filters=dim, kernel_size=5, padding="same", kernel_initializer=k_init)(conv3)
        conv3 = tf.keras.layers.Dropout(dropout, seed=seed)(conv3)
        conv3 = tf.keras.layers.LayerNormalization()(conv3)
        conv3 = tf.keras.layers.ReLU()(conv3)

    out = tf.keras.layers.Concatenate()([conv1,conv2,conv3])
    out = tf.keras.layers.GlobalAveragePooling1D()(out)
    out = tf.keras.layers.Dense(dim, activation='relu', kernel_initializer=k_init)(out)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(out)

    model = tf.keras.models.Model(inputs=inputs, outputs=out, name=f'{dataset}_{arch}_{optimizer}_{dim}_{seed}')
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'], jit_compile=True)
    return model
    
def create_rnn_base_model(dataset, arch, optimizer, dim, seed, dropout, num_classes):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    initializer2 = tf.keras.initializers.HeUniform(seed=seed)
    glorot=tf.keras.initializers.GlorotUniform(seed=seed)
    orthog=tf.keras.initializers.Orthogonal(seed=seed)
    bias = ForgetBiasInitializer()
    
    inputs = layers.Input(input_shape)
    x1 = x2 = x3 = inputs

    for _ in range(2):
        x1 = layers.GRU(dim, return_sequences=True, seed=seed, kernel_initializer=glorot,
                        recurrent_initializer=orthog)(x1)
        x1 = layers.Dropout(dropout, seed=seed)(x1)
        x1 = layers.LayerNormalization()(x1)
        
    for _ in range(2):
        x2 = layers.LSTM(dim, return_sequences=True, seed=seed, kernel_initializer=glorot,
                         recurrent_initializer=orthog, bias_initializer=bias)(x2)
        x2 = layers.Dropout(dropout, seed=seed)(x2)
        x2 = layers.LayerNormalization()(x2)
    out = layers.Concatenate()([x1, x2])
        
    # out = layers.TimeDistributed(layers.Dense(out.shape[-1]//2, activation='relu', kernel_initializer=initializer2))(out)
    out = layers.GlobalAveragePooling1D()(out)
    out = tf.keras.layers.Dense(dim, activation='relu', kernel_initializer=initializer2)(out)
    out = layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer1)(out)

    model = models.Model(inputs, out, name=f'{dataset}_{arch}_{optimizer}_{dim}_{seed}')
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return model

# --- Train Base Models ---
def train_base_models(dataset, archs, optimizers, dim, seeds, dropout, num_classes, batch_size,
                      epochs, sub_folder, X_train, X_val, y_train, y_val):
    for arch in archs:
        for optimizer in optimizers:
            for seed in seeds:
                tf.keras.backend.clear_session()
                np.random.seed(seed)
                tf.random.set_seed(seed)
                random.seed(seed)
                callback = [
                    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
                ]
                model = (create_cnn_base_model(dataset, arch, optimizer, dim, seed, dropout, num_classes) if arch=='cnn'
                         else create_rnn_base_model(dataset, arch, optimizer, dim, seed, dropout, num_classes))
                model.summary()
                print(f'\nTraining {dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}...')
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                          verbose=1, callbacks=callback, class_weight=class_weights_dict)
                model.save(f'test_models/{sub_folder}/{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}.keras')
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
epochs = 10
# batch_size = 512
sub_folder = 'M12'
archs = ['cnn', 'rnn']
wl = 100
dropout = 0.5
num_heads = 2
seeds = [11,5,16,13,25,3,15,1,14]# [6, 28, 42, 95, 138, 196, 276, 496, 8128]  #, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,
         # 233, 377, 610, 987, 1597, 2584 , 4181, 6765,
         # 10946, 17711, 28657, 46368, 75025, 121393,
         # 196418, 317811, 514229]
dim = 128
optimizers = ['adamw'] #,'adam','nadam','adamax']
# stats = [np.mean, np.std, np.ptp, scipy.stats.skew, scipy.stats.kurtosis]

dataset = 'Take5' #,'Quick','Mega','Thunderball','Euro','Powerball','C4L','NYLot','HotPicks']
# for dataset in datasets:
    X_raw = get_real_data(dataset, 120_000)
    data_raw = X_raw.reshape(-1,1)
    
    # data_raw = np.vstack([data_raw, [0]], dtype=np.int8)
    
    extra_features = get_extra_features(data_raw, seeds)
    data = np.hstack([data_raw, extra_features], dtype=np.int8)

    unique_classes = np.unique(X_raw.flatten())
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=X_raw.flatten())
    class_weights_dict = dict(enumerate(class_weights))
    
    del X_raw, data_raw, extra_features
    # data = data_raw
    # data = np.load('data/Take5_stats_features.npy')

    batch_size = compute_batch_size(len(data))
    features = data.shape[-1]
    dim = features*4
    input_shape = (wl, features)
    print(f'\ninput_shape: {input_shape}\n')
    
    X, y = generate_dataset(data, wl, features)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    del X, y, X_temp, y_temp
    
    # Scaling data
    scaler = MinMaxScaler()
    X_train_2d = X_train.reshape(-1, X_train.shape[1]*X_train.shape[2])
    X_val_2d = X_val.reshape(-1, X_val.shape[1]*X_val.shape[2])
    X_test_2d = X_test.reshape(-1, X_test.shape[1]*X_test.shape[2])
    
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_val_scaled = scaler.transform(X_val_2d)
    X_test_scaled = scaler.transform(X_test_2d)

    joblib.dump(scaler, f'test_models/{sub_folder}/{dataset}_base_scaler.joblib')
    
    del X_train_2d, X_val_2d, X_test_2d, scaler
    
    # Reshape back to 3D format
    X_train = X_train_scaled.reshape(-1, X_train.shape[1], X_train.shape[2])
    X_val = X_val_scaled.reshape(-1, X_val.shape[1], X_val.shape[2])
    X_test = X_test_scaled.reshape(-1, X_test.shape[1], X_test.shape[2])

    del X_train_scaled, X_val_scaled, X_test_scaled
    
    train_base_models(dataset, archs, optimizers, dim, seeds, dropout, num_classes, batch_size,
                                    epochs, sub_folder, X_train, X_val, y_train, y_val)

    del X_train, X_val, X_test
