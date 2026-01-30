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

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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


# --- Base Model Generator ---
def create_cnn_base_model(dataset, arch, optimizer, dim, seed, dropout, num_classes):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=11)
    initializer2 = tf.keras.initializers.HeUniform(seed=5)    
    inputs = tf.keras.layers.Input(input_shape)

    out = cnn_net(inputs, dim, dropout, seed)

    out = tf.keras.layers.GlobalAveragePooling1D()(out)
    # out = tf.keras.layers.Dense(dim*2, activation='relu', kernel_initializer=initializer2)(out)
    out = layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer1)(out)

    model = tf.keras.models.Model(inputs=inputs, outputs=out, name=f'{dataset}_{arch}_{optimizer}_{dim}_{seed}')
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


def create_rnn_base_model(dataset, arch, optimizer, dim, seed, dropout, num_classes):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    initializer2 = tf.keras.initializers.HeUniform(seed=seed)
    glorot=tf.keras.initializers.GlorotUniform(seed=seed)
    orthog=tf.keras.initializers.Orthogonal(seed=seed)
    bias = ForgetBiasInitializer()
    
    inputs = layers.Input(input_shape)
    x1 = x2 = x3 = inputs

    for _ in range(1):
        x1 = layers.GRU(dim, return_sequences=True, seed=seed, kernel_initializer=glorot,
                        recurrent_initializer=orthog)(x1)
        x1 = layers.Dropout(dropout, seed=seed)(x1)
        x1 = layers.LayerNormalization()(x1)
        
    for _ in range(1):
        x2 = layers.LSTM(dim, return_sequences=True, seed=seed, kernel_initializer=glorot,
                         recurrent_initializer=orthog, bias_initializer=bias)(x2)
        x2 = layers.Dropout(dropout, seed=seed)(x2)
        x2 = layers.LayerNormalization()(x2)
    out = layers.Add()([x1, x2])
        
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
            if arch=='rnn' or arch=='cnn':
                for seed in seeds:
                    tf.keras.backend.clear_session()
                    callback = [
                        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
                    ]                    
                    model = create_rnn_base_model(dataset, arch, optimizer, dim, seed, dropout, num_classes)
                    # model.summary()
                    print(f'\nTraining {dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}...')
                    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                              verbose=1, callbacks=callback, class_weight=class_weights_dict)
                    model.save(f'test_models/{sub_folder}/{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}.keras')
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
                print(f'\nTraining {dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}...')
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                          verbose=1, callbacks=callback, class_weight=class_weights_dict)
                model.save(f'test_models/{sub_folder}/{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}.keras')
                del model


def load_base_models(sub_folder, dataset, archs, optimizers, dim, seeds, l=0):
    # models = []
    # for dataset in datasets:
    for arch in archs:
        for optimizer in optimizers:
            for seed in seeds:
                if l==0:
                    model = saving.load_model(f'test_models/{sub_folder}/{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}.keras')                        
                else:
                    model = saving.load_model(f'test_models/{sub_folder}/{dataset}{l}_{optimizer}_dim{dim}_seed{seed}.keras')
                # models.append(model)
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
    dim = 64
    num_heads = 4
    x = layers.Dense(dim, activation="relu", kernel_initializer=initializer2)(inputs)
    x = layers.LayerNormalization()(x)

    for _ in range(2):
        # Self-attention over the base model outputs
        mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim//num_heads, seed=seed, kernel_initializer=initializer1)(x,x)
        mha = layers.Dropout(dropout, seed=seed)(mha)
        x = layers.LayerNormalization()(x + mha)
        # FFN
        ffn = layers.Dense(dim*4, activation='relu', kernel_initializer=initializer2)(x)
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
                       X_train, X_val, X_test, y_train, y_val, base='base', l=0, tot_datasets=0):
        
    p_train = []
    p_val = []
    p_test = []

    model = None
    train_model = False
    if base == 'base':
        # dataset = ['Meta_L'] if len(datasets)==1 else dataset
        # seeds = [42] if dataset=='Meta_L' else seeds
        print(f'\nGetting probabilities from all models in {sub_folder}...')
        base_models = load_base_models(sub_folder, dataset, archs, optimizers, dim, seeds, l)  # Get models using generator

        for bm in base_models:
            print(f'\nGetting probabilities from {bm.name} in {sub_folder}...')
            p_train.append(bm.predict(X_train, verbose=0))
            p_val.append(bm.predict(X_val, verbose=0))
            p_test.append(bm.predict(X_test, verbose=0))
            
            loss, acc = bm.evaluate(X_test, y_test, verbose=0)
            print(f'\nModel {bm.name} performance\t| Loss: {loss:.4f}\t| Accuracy: {acc*100:.2f}%')
            
        p_train = np.stack(p_train, axis=1)  # shape: (batch, num_models, num_classes)
        p_val = np.stack(p_val, axis=1)
        p_test = np.stack(p_test, axis=1)
        tot_datasets += 1
        if tot_subs == 1:
            p_train = np.concatenate([p_train1, p_train], axis=1) if p_train1 is not None else p_train
            p_val   = np.concatenate([p_val1, p_val], axis=1)     if p_train1 is not None else p_val
            p_test  = np.concatenate([p_test1, p_test], axis=1)   if p_train1 is not None else p_test
            
            train_model = True if tot_subs==1 else False
            print(f'\nTotal datasets used so far for {sub_folder}: {tot_datasets}\n')

    else:
        p_train = X_train
        p_val = X_val
        p_test = X_test
        tot_datasets = len(datasets)
        train_model = True
        print(f'\nTraining Meta Learners Only. All base datasets and sub_folders done...\n')


    if train_model: # and tot_subs==len(sub_folders):
        mp_train = []
        mp_val   = []
        mp_test  = []        
        l += 1
        num_models = p_train.shape[1]
        for optimizer in ['adamw']:
            for seed in [42]:
                tf.keras.backend.clear_session()
                callback = [
                    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
                ]
                model = create_meta_learner(num_models, num_classes, dim, optimizer, seed, dropout, l)
                model.summary()
                print(f'\nTraining Meta_L{l}_{optimizer}_dim{dim}_seed{seed}...')
                model.fit(
                    p_train, y_train, epochs=100, batch_size=batch_size, validation_data=(p_val, y_val),
                          verbose=1, callbacks=callback, class_weight=class_weights_dict
                )
                model.save(
                    f'test_models/{sub_folder}/Meta_L{l}_{optimizer}_dim{dim}_seed{seed}.keras'
                )                
                mp_train.append(model.predict(p_train, verbose=0))
                mp_val.append(model.predict(p_val, verbose=0))
                mp_test.append(model.predict(p_test, verbose=0))
                                                
        p_train = np.stack(mp_train, axis=1)
        p_val = np.stack(mp_val, axis=1)
        p_test = np.stack(mp_test, axis=1)
        return model, p_train, p_val, p_test
        
    else:
        return model, p_train, p_val, p_test                


# --- Parameters ---
num_classes = 10
epochs = 10
# batch_size = 512
sub_folder = 'M14'
archs = ['cnn','rnn']  #, 'lstm', 'gru', 'cnn']
wl = 10
dropout = 0.5
num_heads = 2
seeds = [11,5,16,13,25,3,15,1,14]  #,4,19,6,12,23,20,8,9,18,37,38,82,83,46,33,49,410,1000]# [6, 28, 42, 95, 138, 196, 276, 496, 8128]  #, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,
         # 233, 377, 610, 987, 1597, 2584 , 4181, 6765,
         # 10946, 17711, 28657, 46368, 75025, 121393,
         # 196418, 317811, 514229]
# dim = 128
optimizers = ['adamw','rmsprop']  #'adam','nadam','adamax']
# stats = [np.mean, np.std, np.ptp, scipy.stats.skew, scipy.stats.kurtosis]

dataset = 'Take5' #,'Quick','Mega','Thunderball','Euro','Powerball','C4L','NYLot','HotPicks']
# for dataset in datasets:
X_raw = get_real_data(dataset, 120_000)

unique_classes = np.unique(X_raw.flatten())
class_weights = compute_class_weight('balanced', classes=unique_classes, y=X_raw.flatten())
class_weights_dict = dict(enumerate(class_weights))

data_raw = X_raw.reshape(-1,1)
data_raw = np.vstack([data_raw[1:], [0]], dtype=np.int8)

# print(f'\nOriginal data_raw tail: {data_raw[-10:].flatten()}\n')

fin_res = []
for _ in range(1):
    # extra_features = get_extra_features(data_raw, seeds)
    # data = np.hstack([data_raw, extra_features], dtype=np.int8)
    data = data_raw
    
    # del X_raw, data_raw, extra_features
    # data = data_raw
    # data = np.load('data/Take5_stats_features.npy')
    
    batch_size = compute_batch_size(len(data))
    features = data.shape[-1]
    input_shape = (wl, features)
    print(f'\ninput_shape: {input_shape}\n')    
    
    dim = features*4

    X, y = generate_dataset(data, wl, features)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    # del X, y, X_temp, y_temp
    
    # Scaling data
    scaler = MinMaxScaler()
    X_train_2d = X_train.reshape(-1, X_train.shape[1]*X_train.shape[2])
    X_val_2d = X_val.reshape(-1, X_val.shape[1]*X_val.shape[2])
    X_test_2d = X_test.reshape(-1, X_test.shape[1]*X_test.shape[2])
    
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_val_scaled = scaler.transform(X_val_2d)
    X_test_scaled = scaler.transform(X_test_2d)
    
    joblib.dump(scaler, f'test_models/{sub_folder}/{dataset}_base_scaler.joblib')
    
    # del X_train_2d, X_val_2d, X_test_2d, scaler
    
    # Reshape back to 3D format
    X_train = X_train_scaled.reshape(-1, X_train.shape[1], X_train.shape[2])
    X_val = X_val_scaled.reshape(-1, X_val.shape[1], X_val.shape[2])
    X_test = X_test_scaled.reshape(-1, X_test.shape[1], X_test.shape[2])
    
    # del X_train_scaled, X_val_scaled, X_test_scaled
    
    train_base_models(dataset, archs, optimizers, dim, seeds, dropout, num_classes, batch_size,
                                    epochs, sub_folder, X_train, X_val, y_train, y_val)
    
    # del X_train, X_val, X_test
    
    
    #############################################################################################
    #############################################################################################
    #############################################################################################
    
    # for _ in range(10):
    tot_subs = 1
    meta1, p_train1, p_val1, p_test1 = None, None, None, None
    # for i, sub_folder in enumerate(sub_folders):
    # extra_features = get_extra_features(data_raw, seeds[i])
    # data = np.hstack([data_raw, extra_features], dtype=np.int8)
    
    # batch_size = compute_batch_size(len(data))
    # features = data.shape[-1]
    # dim = features*4
    # input_shape = (wl, features)
    # print(f'\ninput_shape: {input_shape}\n')
    
    # X, y = generate_dataset(data, wl, features)
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    
    # Scaling data
    # X_train_2d = X_train.reshape(-1, X_train.shape[1]*X_train.shape[2])
    # X_val_2d   = X_val.reshape(-1, X_val.shape[1]*X_val.shape[2])
    # X_test_2d  = X_test.reshape(-1, X_test.shape[1]*X_test.shape[2])

    tot_datasets = 0
    # for dataset in datasets:
    # scaler = joblib.load(f'test_models/{sub_folder}/{dataset}_base_scaler.joblib') #MinMaxScaler()
    # print(f'\nLoaded Scaler: {dataset}_base_scaler in {sub_folder}')
    
    # X_train_scaled = scaler.transform(X_train_2d)
    # X_val_scaled   = scaler.transform(X_val_2d)
    # X_test_scaled  = scaler.transform(X_test_2d)
    
    # # Reshape back to 3D format
    # X_train_reshaped = X_train_scaled.reshape(-1, X_train.shape[1], X_train.shape[2])
    # X_val_reshaped   = X_val_scaled.reshape(-1, X_val.shape[1], X_val.shape[2])
    # X_test_reshaped  = X_test_scaled.reshape(-1, X_test.shape[1], X_test.shape[2])
    
    # # del X_train_scaled, X_val_scaled, X_test_scaled
    
    # unique_classes     = np.unique(X_raw.flatten())
    # class_weights      = compute_class_weight('balanced', classes=unique_classes, y=X_raw.flatten())
    # class_weights_dict = dict(enumerate(class_weights))
    
    # The 1st Meta learner trains on base models
    tf.keras.backend.clear_session()
    meta1, p_train1, p_val1, p_test1 = train_meta_learner(
        archs, num_classes, batch_size, epochs, dataset, sub_folder, optimizers, dim,
        seeds, dropout, X_train, X_val, X_test,
        y_train, y_val, base='base', l=0, tot_datasets=tot_datasets)
        
    tot_subs += 1
    
    print('\nComparing scores for different Meta Levels')
    _ = evaluate_model(meta1, p_test1, 'Meta_L1')

    print('\n\n\t\tStarting test inference using preds as updates to data_raw...\n\n')
    # del (p_train1, p_val1, p_test1, X_train, X_val, X_test, X_train_scaled,
    #      X_val_scaled, X_test_scaled, X_train_reshaped, X_val_reshaped, X_test_reshaped)
    # +---------------------------------------------------------+
    # +--- Test inference using preds as updates to data_raw ---+
    # +---------------------------------------------------------+
    
    '''
        This section of the code as well as optimizations will 
        be sorted later once proof of concept confirmed.
    '''
        
    # updated_data_raw = data_raw  #np.vstack([data_raw, new_pred], dtype=np.int8)
    print(f'\nLast 10 entries of data_raw to be used for inference: {data_raw[-10:].flatten()}')
    
    p_train = []
    # for i, sub_folder in enumerate(sub_folders):
    # extra_features = get_extra_features(updated_data_raw, seeds[i]) if len(updated_data_raw)>len(data_raw) else extra_features
    # data = np.hstack([updated_data_raw, extra_features], dtype=np.int8)         
    
    # batch_size = compute_batch_size(len(data))
    # features = data.shape[-1]
    # dim = features*4
    # input_shape = (wl, features)
    # print(f'\ninput_shape: {input_shape}\n')
    
    # for dataset in datasets:
    # scaler = joblib.load(f'test_models/{sub_folder}/{dataset}_base_scaler.joblib') #MinMaxScaler()
    # print(f'\nLoaded Scaler: {dataset}_base_scaler in {sub_folder}')

    # Reshape data according saved scaler requirements
    pred_data = data[-10:].reshape(-1, wl*features)

    # Scale data
    pred_data_scaled = scaler.transform(pred_data)
    
    # Reshape to 3D format as per model expectation
    pred_data_reshaped = pred_data_scaled.reshape(-1, wl, features)

    tf.keras.backend.clear_session()
    print(f'\nGetting probabilities from all models in {sub_folder}...')
    base_models = load_base_models(sub_folder, dataset, archs, optimizers, dim, seeds, l=0)  # Get models using generator

    for bm in base_models:
        print(f'\nGetting probabilities from {bm.name} in {sub_folder}...')
        bm_pred = bm.predict(pred_data_reshaped, verbose=0)
        p_train.append(bm_pred)
        print(f'Recommendation from {bm.name}: {np.argmax(bm_pred, axis=1).flatten()}')
        del bm_pred
                
    p_train = np.stack(p_train, axis=1)  # shape: (batch, num_models, num_classes)
    
    tf.keras.backend.clear_session()
    new_pred = np.argmax(meta1.predict(p_train, verbose=0), axis=1)
    fin_res.extend(new_pred)
        
    data_raw = np.vstack([data_raw[1:], new_pred], dtype=np.int8)
    print(f'\nLast 10 entries of updated_data_raw: {data_raw[-10:].flatten()}')
    print(f'\n\n\t\tResults so far: {fin_res}\n\n')
    
    
