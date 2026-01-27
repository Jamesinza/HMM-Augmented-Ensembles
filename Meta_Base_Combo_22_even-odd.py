# Silencing annoying warnings
import shutup
shutup.please()

import gc
import math
import random
import joblib
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import skew
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, saving
from keras_hub.layers import FNetEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from hmmlearn.hmm import GaussianHMM, CategoricalHMM, GMMHMM, PoissonHMM, MultinomialHMM
from hmmlearn.vhmm import VariationalGaussianHMM

seed=42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# tf.keras.mixed_precision.set_global_policy('mixed_float16')
# tf.keras.config.enable_unsafe_deserialization()

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# === TSMixer BLOCK ===
@tf.keras.saving.register_keras_serializable()
class TSMixerBlock(tf.keras.layers.Layer):
    def __init__(self, time_steps=10, num_features=1, hidden_dim=64, dropout_rate=0.1, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.time_steps = time_steps
        self.num_features = num_features

    def build(self, input_shape):
        self.time_dense1 = layers.Dense(
            self.hidden_dim, activation='gelu', kernel_initializer=tf.keras.initializers.HeNormal(seed=self.seed))
        self.time_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)
        self.time_dense2 = layers.Dense(self.time_steps,
                                        kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed))
        self.time_norm = layers.LayerNormalization()

        self.feature_dense1 = layers.Dense(
            self.hidden_dim, activation='gelu', kernel_initializer=tf.keras.initializers.HeNormal(seed=self.seed))
        self.feature_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)
        self.feature_dense2 = layers.Dense(self.num_features,
                                           kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed))
        self.feature_norm = layers.LayerNormalization()

        super().build(input_shape)

    def call(self, x):
        # Time-mixing
        residual = x
        x_t = tf.transpose(x, perm=[0, 2, 1])
        x_t = self.time_dense1(x_t)
        x_t = self.time_dropout(x_t)
        x_t = self.time_dense2(x_t)
        x = tf.transpose(x_t, perm=[0, 2, 1])
        x = self.time_norm(x + residual)

        # Feature-mixing
        residual = x
        x_f = self.feature_dense1(x)
        x_f = self.feature_dropout(x_f)
        x_f = self.feature_dense2(x_f)
        x = self.feature_norm(x + residual)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "seed": self.seed,
            "time_steps": self.time_steps,
            "num_features": self.num_features
        })
        return config


@tf.keras.saving.register_keras_serializable()
class TSMixerBlock1(tf.keras.layers.Layer):
    def __init__(self, time_steps=10, num_features=1, hidden_dim=64, dropout_rate=0.1, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.time_steps = time_steps
        self.num_features = num_features

    def build(self, input_shape):
        self.time_dense1 = layers.GRU(self.hidden_dim, return_sequences=True, activation='tanh',
                                      seed=self.seed, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
                                      recurrent_initializer=tf.keras.initializers.Orthogonal(seed=self.seed))
        self.time_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)
        self.time_dense2 = layers.Dense(self.time_steps, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed))
        self.time_norm = layers.LayerNormalization()

        self.feature_dense1 = layers.GRU(self.hidden_dim, return_sequences=True, activation='tanh',
                                         seed=self.seed, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed),
                                         recurrent_initializer=tf.keras.initializers.Orthogonal(seed=self.seed))
        self.feature_dropout = layers.Dropout(self.dropout_rate, seed=self.seed)
        self.feature_dense2 = layers.Dense(self.num_features, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed))
        self.feature_norm = layers.LayerNormalization()

        super().build(input_shape)

    def call(self, x):
        # Time-mixing
        residual = x
        x_t = tf.transpose(x, perm=[0, 2, 1])
        x_t = self.time_dense1(x_t)
        x_t = self.time_dropout(x_t)
        x_t = self.time_dense2(x_t)
        x = tf.transpose(x_t, perm=[0, 2, 1])
        x = self.time_norm(x + residual)

        # Feature-mixing
        residual = x
        x_f = self.feature_dense1(x)
        x_f = self.feature_dropout(x_f)
        x_f = self.feature_dense2(x_f)
        x = self.feature_norm(x + residual)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "seed": self.seed,
            "time_steps": self.time_steps,
            "num_features": self.num_features
        })
        return config        
 

def select_best_n_components(X, component_range):
    best_score = -np.inf
    best_n_components = None
    for n in range(component_range):
        model = GaussianHMM(n_components=n, covariance_type='full', n_iter=100)
        try:
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_n_components = n
                print(f"\nBest so far n_components = {best_n_components}")
        except:
            continue
    return best_n_components    


def compute_window_stats(data, wl):
    # X_windows: shape (num_samples, window_size)
    features = np.empty([len(data) - wl, 4], dtype=np.float32)
    features = []
    for i in range(len(data) - wl):
        window = data[i:i+wl]
        mean = np.mean(window)
        std = np.std(window)
        skewness = skew(window)
        kurt = kurtosis(window)
        stats = np.hstack([mean, std, skewness, kurt])
        features[i] = stats
    return features    


def transform_selector(data, sub_folder, dataset):
    s = skew(data.flatten())
    
    if s > 0.5:
        tf = PowerTransformer(method='yeo-johnson')
    elif s < -0.5:
        tf = PowerTransformer(method='yeo-johnson')
    else:
        tf = StandardScaler()
    
    tf_data = tf.fit_transform(data)
    joblib.dump(tf, f'test_models/{sub_folder}/{dataset}_base_scaler.joblib')
    
    return tf_data.astype(np.float32), tf
    

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


# --- Generate 3D data sequences ---
def generate_dataset(X, data, wl=10, features=1):
    X_test = np.empty([len(data)-wl, wl, features], dtype=np.float32)
    y_test = np.empty([len(data)-wl, 1], dtype=np.int8)
    for i in range(len(data)-wl):
        X_test[i] = X[i:i+wl]
        y_test[i] = data[i+wl]    
    return X_test, y_test


# --- Generate 2D data sequences ---
def create_windows(data, wl=10):
    X_test = np.empty([len(data)-wl, wl], dtype=np.int8)
    y_test = np.empty([len(data)-wl, 1], dtype=np.int8)
    for i in range(len(data) - wl):
        X_test[i] = data[i:i + wl]
        y_test[i] = data[i + wl]
    return X_test, y_test


@tf.keras.saving.register_keras_serializable()
class ForgetBiasInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        # Bias layout for LSTM: [input_gate, forget_gate, cell_gate, output_gate]
        result = tf.zeros(shape, dtype=dtype)
        n = shape[0] // 4
        result = tf.tensor_scatter_nd_update(result, [[n]], [1.0])
        return result


def hmm_normal_stack(X_raw, rngs, n=5):
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
        a=b=c=d=e=n
        for _ in range(1):
            # GaussianHMM and CategoricalHMM
            # print('\nNow running GaussianHMM normal...')
            hmm_g = GaussianHMM(n_components=5, covariance_type="full", random_state=rng)
            # print('\nNow running CategoricalHMM normal...')
            # hmm_c = CategoricalHMM(n_components=b, n_features=10, random_state=rng)
            hmm_g.fit(hs1)
            # hmm_c.fit(hs2)
            hs1_pred = hmm_g.predict(hs1).reshape(-1, 1).astype(np.float32)
            # a = len(np.unique(hs1_pred))
            # print(f'\na: {a}')
            # hs2_pred = hmm_c.predict(hs2).reshape(-1, 1).astype(np.int8)
            # b = len(np.unique(hs2_pred))
            # print(f'\nb: {b}')
            base_features = (np.hstack([base_features, hs1_pred], dtype=np.float32)
                             if base_features is not None
                             else hs1_pred)
            hs1 = hs1_pred  # Update for potential further iterations
            # print(f'\nhs1:\n{hs1}')
            # print(f'hs2:\n{hs2}\n')

        # for i in range(1):
        #     # GMMHMM Feature
        #     # print('\nNow running GMMHMM stepped...')
        #     hmm_gmm = GMMHMM(n_components=5, n_mix=1, covariance_type="full", random_state=rng)
        #     hmm_gmm.fit(hs3)
        #     hs3_pred = hmm_gmm.predict(hs3).reshape(-1, 1).astype(np.float32)
        #     c = len(np.unique(hs3_pred))
        #     # print(f'\nc: {c}')
        #     base_features = (np.hstack([base_features, hs3_pred], dtype=np.float32)
        #                      if base_features is not None
        #                      else hs3_pred)
        #     hs3 = hs3_pred  # Update for potential further iterations
        #     # print(f'\nhs3:\n{hs3}')

        # for _ in range(1):
        #     # PoissonHMM Feature
        #     # print('\nNow running PoissonHMM normal...')
        #     hmm_pois = PoissonHMM(n_components=d, random_state=rng)
        #     hmm_pois.fit(hs4)
        #     hs4_pred = hmm_pois.predict(hs4).reshape(-1, 1).astype(np.float32)
        #     d = len(np.unique(hs4_pred))
        #     # print(f'\nc: {c}')
        #     base_features = (np.hstack([base_features, hs4_pred], dtype=np.float32)
        #                      if base_features is not None
        #                      else hs4_pred)
        #     hs4 = hs4_pred  # Update for potential further iterations
        #     # print(f'\nhs4:\n{hs4}')

        # for _ in range(1):
        #     # VariationalGaussianHMM Feature
        #     # print('\nNow running VariationalGaussianHMM normal...')
        #     hmm_multi = VariationalGaussianHMM(n_components=9, random_state=rng)
        #     hmm_multi.fit(hs5)
        #     hs5_pred = hmm_multi.predict(hs5).reshape(-1, 1).astype(np.float32)
        #     # print(hs5_pred)
        #     # e = len(np.unique(hs5_pred))
        #     # print(f'\ne: {e}')
        #     base_features = (np.hstack([base_features, hs5_pred], dtype=np.float32)
        #                      if base_features is not None
        #                      else hs5_pred)
        #     hs5 = hs5_pred  # Update for potential further iterations
        #     # print(f'\nhs5:\n{hs5}')
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
        a=b=2
        for i in range(1):
            # GaussianHMM and CategoricalHMM
            # print('\nNow running GaussianHMM stepped...')
            hmm_g = GaussianHMM(n_components=a - i, covariance_type="full", random_state=rng)
            # print('\nNow running CategoricalHMM stepped...')
            # hmm_c = CategoricalHMM(n_components=b - i, n_features=10, random_state=rng)
            hmm_g.fit(hs1)
            # hmm_c.fit(hs2)
            hs1_pred = hmm_g.predict(hs1).reshape(-1, 1).astype(np.float32)
            a = len(np.unique(hs1_pred))
            # print(f'\na: {a}')
            # hs2_pred = hmm_c.predict(hs2).reshape(-1, 1).astype(np.int8)
            # b = len(np.unique(hs2_pred))
            # print(f'\nb: {b}')
            base_features = (np.hstack([base_features, hs1_pred], dtype=np.float32)
                             if base_features is not None
                             else np.hstack([hs1_pred], dtype=np.float32))
            hs1 = hs1_pred  # Update for potential further iterations
            # print(f'\nhs1:\n{hs1}')
            # print(f'hs2:\n{hs2}\n')
    return base_features


def get_extra_features(X_raw, rng):
    """
    Create HMM-based and NVG-based features.
    """
    data = hmm_normal_stack(X_raw, rng)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    # joblib.dump(scaler, f'test_models/{sub_folder}/{dataset}_feature_scaler.joblib')
    return scaled_data

   
def compute_batch_size(dataset_length):
    base_unit  = 25000
    base_batch = 32
    batch_size = base_batch * math.ceil(dataset_length / base_unit)
    return batch_size    


def cnn_net(inputs, dim, dropout, seed):
    # k_init = tf.keras.initializers.HeUniform()
    conv1=conv2=conv3=inputs

    conv1 = tf.keras.layers.Conv1D(filters=dim, kernel_size=2, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(conv1)
    conv1 = tf.keras.layers.Dropout(dropout, seed=seed)(conv1)
    conv1 = tf.keras.layers.LayerNormalization()(conv1)
    conv1 = tf.keras.layers.ReLU()(conv1)

    conv2 = tf.keras.layers.Conv1D(filters=dim, kernel_size=3, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(conv2)
    conv2 = tf.keras.layers.Dropout(dropout, seed=seed)(conv2)
    conv2 = tf.keras.layers.LayerNormalization()(conv2)
    conv2 = tf.keras.layers.ReLU()(conv2)

    conv3 = tf.keras.layers.Conv1D(filters=dim, kernel_size=5, padding="same", kernel_initializer=tf.keras.initializers.HeUniform(seed=seed))(conv3)
    conv3 = tf.keras.layers.Dropout(dropout, seed=seed)(conv3)
    conv3 = tf.keras.layers.LayerNormalization()(conv3)
    conv3 = tf.keras.layers.ReLU()(conv3)

    out = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
    return out


def gru_net(inputs, dim, dropout, seeds):
    bias = ForgetBiasInitializer()    
    x1=x2=x3=inputs

    x1 = layers.GRU(dim, return_sequences=True, seed=seed, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[0]),
                    recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seeds[0]))(x1)
    x1 = layers.Dropout(dropout, seed=seeds[0])(x1)
    x1 = layers.LayerNormalization()(x1)

    x2 = layers.GRU(dim, return_sequences=True, seed=seed, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[1]),
                    recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seeds[1]))(x2)
    x2 = layers.Dropout(dropout, seed=seeds[1])(x2)
    x2 = layers.LayerNormalization()(x2)

    x3 = layers.GRU(dim, return_sequences=True, seed=seed, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[2]),
                    recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seeds[2]))(x3)
    x3 = layers.Dropout(dropout, seed=seeds[2])(x3)
    x3 = layers.LayerNormalization()(x3)
    
    out = tf.keras.layers.Concatenate()([x1, x2, x3])
    return out


def lstm_net(inputs, dim, dropout, seeds):
    # glorot=tf.keras.initializers.GlorotUniform()
    # orthog=tf.keras.initializers.Orthogonal()
    bias = ForgetBiasInitializer()    
    x1=x2=x3=inputs

    x1 = layers.LSTM(dim, return_sequences=True, seed=seed, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[0]),
                     recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seeds[0]), bias_initializer=bias)(x1)
    x1 = layers.Dropout(dropout, seed=seeds[0])(x1)
    x1 = layers.LayerNormalization()(x1)

    x2 = layers.LSTM(dim, return_sequences=True, seed=seed, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[1]),
                     recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seeds[1]), bias_initializer=bias)(x2)
    x2 = layers.Dropout(dropout, seed=seeds[1])(x2)
    x2 = layers.LayerNormalization()(x2)

    x3 = layers.LSTM(dim, return_sequences=True, seed=seed, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seeds[2]),
                     recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seeds[2]), bias_initializer=bias)(x3)
    x3 = layers.Dropout(dropout, seed=seeds[2])(x3)
    x3 = layers.LayerNormalization()(x3)
    return x1, x2, x3


def fnet_tsm_block(x, fnet_layers, tsm_layers, wl, features, dim, dropout, seed):
    for _ in range(fnet_layers):
        x = FNetEncoder(dim)(x)
    for _ in range(tsm_layers):
        x = TSMixerBlock(wl, features, dim, dropout, seed)(x)
    return x

    
# --- TSMixer Model ---
def create_tsmixer_base_model(dataset, arch, optimizer, dim, seed, dropout, num_classes, s):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    inputs = layers.Input(shape=input_shape)
    x = inputs
    x = layers.Dense(dim, activation='gelu', kernel_initializer=initializer1)(x)

    for _ in range(all_layers):
        x1 = x
        x1 = fnet_tsm_block(x1, fnet_layers, tsm_layers, wl, features, dim, dropout, seed)
        x = layers.Add()([x1, x])
        # x = layers.LayerNormalization()(x)
        
    # x = layers.GRU(dim, return_sequences=True, seed=seed, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
    #                     recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seed))(x)
    # x = layers.Dense(dim, activation='gelu', kernel_initializer=initializer1)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dim, activation='gelu', kernel_initializer=initializer1)(x)
    out = layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer1)(x)
    model = models.Model(inputs, out, name=f'{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}')
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'], jit_compile=True)
    return model


# --- Base Model Generator ---
def create_cnn_base_model(dataset, arch, optimizer, dim, seed, dropout, num_classes, s):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    initializer2 = tf.keras.initializers.HeUniform(seed=seed)    
    inputs = tf.keras.layers.Input(input_shape)

    out = cnn_net(inputs, dim, dropout, seed)

    out = tf.keras.layers.GlobalAveragePooling1D()(out)
    out = tf.keras.layers.Dense(dim, activation='relu', kernel_initializer=initializer2)(out)
    out = layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer1)(out)

    model = tf.keras.models.Model(inputs=inputs, outputs=out, name=f'{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}')
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'], jit_compile=True)
    return model    

    
def create_gru_base_model(dataset, arch, optimizer, dim, seeds, dropout=0.5, num_classes=10):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=11)
    initializer2 = tf.keras.initializers.HeUniform(seed=5)
    dropout=0.5
    inputs = layers.Input(input_shape)

    # gru1, gru2, gru3 = gru_net(inputs, dim, dropout, seeds[:3])
    # gru4, gru5, gru6 = gru_net(inputs, dim, dropout, seeds[3:6])
    # gru7, gru8, gru9 = gru_net(inputs, dim, dropout, seeds[6:9])
    # all_gru = [gru1, gru2, gru3, gru4, gru5, gru6, gru7, gru8, gru9]
    
    # out = tf.keras.layers.Add()(all_gru)
    seed = 16
    x1 = layers.GRU(dim, return_sequences=True, seed=seed, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                    recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seed))(inputs)    
    for seed in seeds:
        x = layers.GRU(dim, return_sequences=True, seed=seed, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                         recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seed))(x1)
        x = layers.Dropout(dropout, seed=seed)(x)
        x = layers.Add()([x1, x])
        x1 = layers.LayerNormalization()(x)    
    # out = tf.keras.layers.Dense(dim*2, activation='relu', kernel_initializer=initializer2)(out)
    out = layers.GlobalAveragePooling1D()(x1)
    
    out = layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer1)(out)
    seed = 42
    model = models.Model(inputs, out, name=f'{dataset}_{arch}_{optimizer}_{dim}_{seed}')
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return model


def create_lstm_base_model(dataset, arch, optimizer, dim, seeds, dropout=0.6, num_classes=10):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=11)
    initializer2 = tf.keras.initializers.HeUniform(seed=5)
    dropout=0.6
    inputs = layers.Input(input_shape)

    # lstm1, lstm2, lstm3 = lstm_net(inputs, dim, dropout, seeds[:3])
    # lstm4, lstm5, lstm6 = lstm_net(inputs, dim, dropout, seeds[3:6])
    # lstm7, lstm8, lstm9 = lstm_net(inputs, dim, dropout, seeds[6:9])
    # all_lstm = [lstm1, lstm2, lstm3, lstm4, lstm5, lstm6, lstm7, lstm8, lstm9]
    
    # out = tf.keras.layers.Add()(all_lstm)
    # x1 = layers.GRU(dim, return_sequences=True)(inputs)
    seed = 16
    x1 = layers.GRU(dim, return_sequences=True, seed=seed, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                    recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seed))(inputs)     
    for seed in seeds:
        x = layers.LSTM(dim, return_sequences=True, seed=seed, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                         recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seed), bias_initializer=ForgetBiasInitializer())(x1)
        x = layers.Dropout(dropout, seed=seed)(x)
        x = layers.Add()([x1, x])
        x1 = layers.LayerNormalization()(x)
    # out = tf.keras.layers.Dense(dim//2, activation='relu', kernel_initializer=initializer2)(out)
    out = layers.GlobalAveragePooling1D()(x1)
    
    out = layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer1)(out)
    seed = 42
    model = models.Model(inputs, out, name=f'{dataset}_{arch}_{optimizer}_{dim}_{seed}')
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return model


def create_rnn_base_model(dataset, arch, optimizer, dim, seed, dropout, num_classes, s):
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
    out = layers.Add()([x1, x2])
        
    # out = layers.TimeDistributed(layers.Dense(out.shape[-1]//2, activation='relu', kernel_initializer=initializer2))(out)
    out = layers.GlobalAveragePooling1D()(out)
    out = tf.keras.layers.Dense(dim, activation='relu', kernel_initializer=initializer2)(out)
    out = layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer1)(out)

    model = models.Model(inputs, out, name=f'{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}')
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return model    


# --- Train Base Models ---
def train_base_models(dataset, archs, optimizers, dim, seeds, dropout, num_classes, batch_size,
                      epochs, sub_folder, X_train, X_val, y_train, y_val, s):
    for arch in archs:
        for optimizer in optimizers:
            if arch=='rnn' or arch=='cnn' or arch=='tsmixer':
                for seed in seeds:
                    tf.keras.backend.clear_session()
                    callback = [
                        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
                    ]
                    if arch=='rnn':
                        model = create_rnn_base_model(dataset, arch, optimizer, dim, seed, dropout, num_classes, s)
                    elif arch=='cnn':
                        model = create_cnn_base_model(dataset, arch, optimizer, dim, seed, dropout, num_classes, s)
                    else:
                        model = create_tsmixer_base_model(dataset, arch, optimizer, dim, seed, dropout, num_classes, s)
                    model.summary()
                    print(f'\nTraining {dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}...')
                    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                              verbose=1, callbacks=callback, class_weight=class_weights_dict)
                    model.save(f'test_models/{sub_folder}/{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}.keras')
                    del model                
            else:
                tf.keras.backend.clear_session()
                callback = [
                    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
                ]
                
                # if arch=='cnn':
                #     model = create_cnn_base_model(dataset, arch, optimizer, dim, seeds, dropout, num_classes)
                    
                if arch=='gru':
                    model = create_gru_base_model(dataset, arch, optimizer, dim, seeds, dropout, num_classes)
                    
                else:
                    model = create_lstm_base_model(dataset, arch, optimizer, dim, seeds, dropout, num_classes)
                
                # model.summary()
                seed = 42
                print(f'\nTraining {dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}...')
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                          verbose=1, callbacks=callback, class_weight=class_weights_dict)
                model.save(f'test_models/{sub_folder}/{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}.keras')
                del model


def load_base_models(sub_folder, dataset, archs, optimizers, dim, seeds, l=0, t_dim=128):
    # models = []
    # for dataset in datasets:
    for s in range(div):
        for arch in archs:
            for optimizer in optimizers:
                if l==0:
                    for seed in seeds:
                        model = saving.load_model(f'test_models/{sub_folder}/{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}.keras')
                        yield model
                else:
                    for seed in [42]:
                        model = saving.load_model(f'test_models/{sub_folder}/{dataset}{l}_{optimizer}_dim{t_dim}_seed{seed}.keras')
                        yield model


# --- Evaluation ---
def evaluate_model(model, X_test, name):
    pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, pred)
    print(f'\nLast 10 results for {model.name}...')
    print(f'Predictions: {pred[-10:]}')
    print(f'Real Values: {y_test[-10:].flatten()}')
    print(f'{name} Accuracy: {acc:.4f}')
    return pred[-1].flatten() 

    
# --- Meta Model Generator ---
def create_meta_learner(num_models, num_classes, dim, optimizer, seed, dropout, l, num_heads=4):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    initializer2 = tf.keras.initializers.HeNormal(seed=seed)
    inputs = layers.Input(shape=(num_models, num_classes))
    num_heads = 4
    x = layers.Dense(dim, activation="gelu", kernel_initializer=initializer2)(inputs)
    x = layers.LayerNormalization()(x)

    for _ in range(2):
        # Self-attention over the base model outputs
        mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim//num_heads, seed=seed, kernel_initializer=initializer1)(x,x)
        mha = layers.Dropout(dropout, seed=seed)(mha)
        x = layers.LayerNormalization()(x + mha)
        # FFN
        ffn = layers.Dense(dim*4, activation='gelu', kernel_initializer=initializer2)(x)
        ffn = layers.Dropout(dropout, seed=seed)(ffn)
        ffn = layers.Dense(dim, kernel_initializer=initializer1)(ffn)
        ffn = layers.Dropout(dropout, seed=seed)(ffn)
        x = layers.LayerNormalization()(x + ffn)

    out = layers.GlobalAveragePooling1D()(x)  # shape: (batch, 64)
    out = layers.Dense(dim // 2, activation='relu', kernel_initializer=initializer2)(out)
    output = layers.Dense(num_classes, activation="softmax", kernel_initializer=initializer1)(out)
    model = models.Model(inputs, output, name=f'Meta_L{l}_{optimizer}_dim{dim}_seed{seed}')
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    return model


# --- Train Meta Models ---
def train_meta_learner(archs, num_classes, batch_size,
                      epochs, dataset, sub_folder, optimizers, dim, seeds, dropout,
                       X_train, X_val, y_train, y_val, base='base', l=0, tot_datasets=0, t_dim=64):
        
    p_train = []
    p_val = []
    # p_test = []

    model = None
    train_model = False
    if base == 'base':
        # dataset = ['Meta_L'] if len(datasets)==1 else dataset
        # seeds = [42] if dataset=='Meta_L' else seeds
        print(f'\nGetting probabilities from all models in {sub_folder}...')
        # tf.keras.config.enable_unsafe_deserialization()
        base_models = load_base_models(sub_folder, dataset, archs, optimizers, dim, seeds, l, t_dim)  # Get models using generator

        for bm in base_models:
            p_train.append(bm.predict(X_train, verbose=0))
            p_val.append(bm.predict(X_val, verbose=0))
            # p_test.append(bm.predict(X_test, verbose=0))
            
            loss, acc = bm.evaluate(X_val, y_val, verbose=0)
            print(f'\nGetting probabilities from {bm.name} in {sub_folder} to train Meta Learner...\t| Loss: {loss:.4f}\t| Accuracy: {acc*100:.2f}%')
            
        p_train = np.stack(p_train, axis=1)  # shape: (batch, num_models, num_classes)
        p_val = np.stack(p_val, axis=1)
        # p_test = np.stack(p_test, axis=1)
        tot_datasets += 1
        if tot_subs == 1:
            p_train = np.concatenate([p_train1, p_train], axis=1) if p_train1 is not None else p_train
            p_val   = np.concatenate([p_val1, p_val], axis=1)     if p_train1 is not None else p_val
            # p_test  = np.concatenate([p_test1, p_test], axis=1)   if p_train1 is not None else p_test
            
            train_model = True if tot_subs==1 else False
            print(f'\nTotal datasets used so far for {sub_folder}: {tot_datasets}\n')

    else:
        p_train = X_train
        p_val = X_val
        # p_test = X_test
        tot_datasets = len(datasets)
        train_model = True
        print(f'\nTraining Meta Learners Only. All base datasets and sub_folders done...\n')


    if train_model: # and tot_subs==len(sub_folders):
        mp_train = []
        mp_val   = []
        # mp_test  = []        
        l += 1
        num_models = p_train.shape[1]
        for optimizer in ['adamw']:
            for seed in [42]:
                tf.keras.backend.clear_session()
                callback = [
                    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
                ]
                model = create_meta_learner(num_models, num_classes, t_dim, optimizer, seed, dropout, l)
                model.summary()
                print(f'\nTraining Meta_L{l}_{optimizer}_dim{t_dim}_seed{seed}...')
                model.fit(
                    p_train, y_train, epochs=100, batch_size=batch_size, validation_data=(p_val, y_val),
                          verbose=1, callbacks=callback, class_weight=class_weights_dict
                )
                model.save(
                    f'test_models/{sub_folder}/Meta_L{l}_{optimizer}_dim{t_dim}_seed{seed}.keras'
                )                
                mp_train.append(model.predict(p_train, verbose=0))
                mp_val.append(model.predict(p_val, verbose=0))
                # mp_test.append(model.predict(p_test, verbose=0))
                                                
        p_train = np.stack(mp_train, axis=1)
        p_val = np.stack(mp_val, axis=1)
        # p_test = np.stack(mp_test, axis=1)
        return model, p_train, p_val
        
    else:
        return model, p_train, p_val


# --- Parameters ---
# num_classes = 10
epochs = 1000
# batch_size = 512
sub_folder = 'M22_even-odd'
archs = ['tsmixer']  #'cnn','rnn']  #, 'lstm', 'gru', 'cnn']
wl = 8
dropout = 0.5
num_heads = 2
seeds = [42] #, 276]  #, 6, 28]  #, 42, 138, 276]
optimizers = ['adamw','rmsprop','adam','nadam','adamax']
dataset = 'Take5'

all_layers = 4
fnet_layers = 2
tsm_layers = 6

# for dataset in datasets:
X_raw = get_real_data(dataset, 120_000)

# unique_classes = np.unique(X_raw.flatten())
# class_weights = compute_class_weight('balanced', classes=unique_classes, y=X_raw.flatten())
# class_weights_dict = dict(enumerate(class_weights))
# num_classes = len(unique_classes)

data_raw = X_raw.reshape(-1,1)
# data_raw = np.vstack([data_raw, [0]], dtype=np.int8)

# div = 1
# splits = len(data_raw)//div

# dim = 128
t_dim = 256

# data = data_raw.copy()

X_odd = data_raw[::2]
X_even  = data_raw[1::2]


data = X_odd
div = 10
splits = len(data)//div

unique_classes = np.unique(data.flatten())
class_weights = compute_class_weight('balanced', classes=unique_classes, y=data.flatten())
class_weights_dict = dict(enumerate(class_weights))
num_classes = len(unique_classes)


fin_res = []
for _ in range(5):
    # Data Transformations
    X_full, data_scaler = transform_selector(data, sub_folder, dataset)

    s = 0  #for s in range(div):
    div = 1
    print(f'\n\n\t\tNow running subset {s}...\n\n')
    X_sub = X_full[-splits*(s+1):-splits*s] if s>0 else X_full[-splits*(s+1):]
    y_sub = data[-splits*(s+1):-splits*s] if s>0 else data[-splits*(s+1):]

    # extra_features = get_extra_features(X_sub, [1, # 2, 3, 5, 8, 13, 21, # 34, 55, 89, 144, 233, 377, 610, 987,
    #                                              # 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393,
    #                                              # 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887,
    #                                              # 9227465, 14930352, 24157817, 39088169, 63245986, 102334155, 165580141,
    #                                              # 267914296, 433494437, 701408733, 1134903170, 1836311903, 2971215073
    #                                             ])
    # X_sub = np.hstack([X_sub, extra_features], dtype=np.float32)
    features = X_sub.shape[-1]
    print(f'\nNumber of new features plus original data for subset: {features}\n')
    input_shape = (wl, features)
    dim = 16 #features*4        
    
    batch_size = compute_batch_size(len(X_sub))
    print(f'\nLength of training data: {len(X_sub)}\n')
    X, y = generate_dataset(X_sub, y_sub, wl, features)

    # Reshape X & y to 3D as per model expectation
    # X = X.reshape(-1, wl, features)
    # y = y.reshape(-1, 1)
    
    # Data splitting
    shifting = 0
    X_train, X_val, y_train, y_val = train_test_split(X[:-shifting] if shifting>0 else X, y[shifting:], test_size=0.1, shuffle=False)
    
    # train_base_models(dataset, archs, optimizers, dim, seeds, dropout, num_classes, batch_size,
    #                                 epochs, sub_folder, X_train, X_val, y_train, y_val, s)

        # del X_sub, y_sub, X, y, X_train, X_val, y_train, y_val

    # extra_features = get_extra_features(X_full, [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
    #                                              1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393,
    #                                              196418, 317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887,
    #                                              9227465, 14930352, 24157817, 39088169, 63245986, 102334155, 165580141,
    #                                              267914296, 433494437, 701408733, 1134903170, 1836311903, 2971215073
    #                                             ])
    # X_full = np.hstack([X_full, extra_features], dtype=np.float32)
    
    # features = X_full.shape[-1]
    # print(f'\nNumber of new features plus original data for full dataset: {features}\n')
    # input_shape = (wl, features)
    # dim = 128 #features*4    

    # print(f'\ninput_shape: {input_shape}\n')     
    # X, y = generate_dataset(X_full, data, wl, features)
    # batch_size = compute_batch_size(len(X))

    # Data splitting
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, shuffle=False)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)    
    
    # for _ in range(10):
    tot_subs = 1
    meta1, p_train1, p_val1 = None, None, None
    tot_datasets = 0
    
    # The 1st Meta learner trains on base models
    tf.keras.backend.clear_session()
    # meta1, p_train1, p_val1 = train_meta_learner(
    #     archs, num_classes, batch_size, epochs, dataset, sub_folder, optimizers, dim,
    #     seeds, dropout, X_train, X_val, y_train, y_val,
    #     base='base', l=0, tot_datasets=tot_datasets, t_dim=t_dim)

    meta1 = load_base_models(sub_folder, 'Meta_L', archs, ['adamw'], 256, [42], l=1)  # Get models using generator
            
    tot_subs += 1
    del meta1, p_train1, p_val1, X_train, X_val, y_train, y_val

    print('\n\n\t\tStarting test inference using preds as updates to data_raw...\n\n')

    # +---------------------------------------------------------+
    # +--- Test inference using preds as updates to data_raw ---+
    # +---------------------------------------------------------+
    
    '''
        This section of the code as well as optimizations will 
        be sorted later once proof of concept confirmed.
    '''
        
    # updated_data_raw = data_raw  #np.vstack([data_raw, new_pred], dtype=np.int8)
    print(f'\nLast {wl} entries of data to be used for inference: {data[-wl:].flatten()}')
    
    p_train = []

    # Reshape data according saved scaler requirements
    pred_data = X_sub[-wl:]  #.reshape(-1, wl*features)
    
    # Reshape to 3D format as per model expectation
    pred_data_reshaped = pred_data.reshape(-1, wl, features)
    del pred_data
    
    tf.keras.backend.clear_session()
    # tf.keras.config.enable_unsafe_deserialization()
    print(f'\nGetting probabilities from all models in {sub_folder}...')
    base_models = load_base_models(sub_folder, dataset, archs, optimizers, dim, seeds, l=0)  # Get models using generator

    for bm in base_models:
        print(f'\nGetting probabilities from {bm.name} in {sub_folder}...')
        bm_pred = bm.predict(pred_data_reshaped, verbose=0)
        p_train.append(bm_pred)
        print(f'Recommendation from {bm.name}: {np.argmax(bm_pred, axis=1).flatten()}')
        del bm_pred
    del pred_data_reshaped            
    
    p_train = np.stack(p_train, axis=1)  # shape: (batch, num_models, num_classes)
    
    tf.keras.backend.clear_session()
    meta1 = saving.load_model(f'test_models/{sub_folder}/Meta_L1_adamw_dim{t_dim}_seed42.keras')
    new_pred = np.argmax(meta1.predict(p_train, verbose=0), axis=1)
    fin_res.extend(new_pred)
    del p_train, meta1
    
    data = np.vstack([data, new_pred], dtype=np.int8)
    print(f'\nLast {wl} entries of updated data: {data[-wl:].flatten()}')
    print(f'\n\n\t\tResults so far: {fin_res}\n\n')
    del new_pred

    