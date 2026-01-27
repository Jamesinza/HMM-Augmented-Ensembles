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
from collections import defaultdict, Counter
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
# from memory_profiler import profile

# import os, psutil, gc, time
# process = psutil.Process(os.getpid())
# def print_mem(msg=""):
#     mem_mb = process.memory_info().rss / (1024 ** 2)
#     print(f"[MEMORY] {msg} RSS: {mem_mb:.2f} MB")
    



# print_mem("Environment setup complete")


def load_base_models(sub_folder, datasets, arch, optimizers, dim, seeds):
    for dataset in datasets:
        print(f'\nGetting probabilities from all models in {sub_folder} for dataset {dataset}...\n')
        for optimizer in optimizers:
            for seed in seeds:
                path  = f'test_models/{sub_folder}/{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}.keras'
                model = saving.load_model(path, compile=False)
                yield model



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


def compute_batch_size(dataset_length):
    base_unit  = 25000
    base_batch = 32
    batch_size = base_batch * math.ceil(dataset_length / base_unit)
    return batch_size    
    

# ==================================================================================================
# ===================================== Training Base Models =======================================
# ==================================================================================================

# --- Prepare datasets using tf.data pipeline
def create_dataset(sequence, W, batch_size, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(sequence)
    dataset = dataset.window(W + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(W + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1, 0]))
    if shuffle==True:
        dataset = dataset.batch(batch_size).shuffle(1000).repeat().prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    return dataset    


# Prepare windowed training data for adaptation:
def prepare_for_adapt(sequence, W):
    dataset = tf.data.Dataset.from_tensor_slices(sequence)
    dataset = dataset.window(W + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(W + 1))
    dataset = dataset.map(lambda window: window[:-1])  # Only inputs
    dataset = dataset.batch(512)  # Batch for efficient adaptation
    return dataset    


def fnet_tsm_block(x, wl, dim, dropout, features, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)    
    for _ in range(2):
        x = FNetEncoder(dim)(x)
    for _ in range(6):
        x = TSMixerBlock(wl, features, dim, dropout, seed)(x)
    return x

    
# --- TSMixer Model ---
def create_tsmixer_base_model(norm, input_shape, wl, norm, dataset, arch, optimizer,
                              dim, seed, dropout, num_classes, features, s):
    initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
    inputs = layers.Input(shape=input_shape)
    # x = inputs
    x = norm(inputs)
    x = layers.Dense(dim, activation='gelu', kernel_initializer=initializer1)(x)

    for _ in range(1):
        x1 = x
        x1 = fnet_tsm_block(x1, wl, dim, dropout, features, seed)
        x = layers.Dropout(dropout)(x)
        x = layers.Add()([x1, x])

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dim, activation='gelu', kernel_initializer=initializer1)(x)
    out = layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer1)(x)
    model = models.Model(inputs, out, name=f'{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}')
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return model


# --- Train Base Models ---
# @profile
def train_base_models(norm, train_ds, val_ds, dataset, arch, optimizers,
                      dim, seeds, dropout, epochs, sub_folder, s, num_classes, class_weights_dict):
    for optimizer in optimizers:
        tf.keras.backend.clear_session()
        for seed in seeds:
            tf.keras.backend.clear_session()
            callback = [
                callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
            ]
            model = create_tsmixer_base_model(norm, input_shape, wl, norm, dataset, arch,
                                              optimizer, dim, seed, dropout, num_classes, features, s)
            # model.summary()
            
            print(f'\nTraining {dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}...')
            model.fit(train_ds, epochs=epochs, batch_size=batch_size, validation_data=val_ds,
                      verbose=1, callbacks=callback, class_weight=class_weights_dict,
                      steps_per_epoch=train_steps, validation_steps=val_steps)
            model.save(f'test_models/{sub_folder}/{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}.keras')
            
            del model, train_ds, val_ds, adapt_ds, norm, callback, batch_size, train_steps, val_steps, seed, history, a, train_data, val_data


def process_data(train_data, val_data, wl, batch_size):
    train_ds = create_dataset(train_data, wl, batch_size, shuffle=False)
    val_ds = create_dataset(val_data, wl, batch_size, shuffle=False)
    adapt_ds = prepare_for_adapt(train_data, wl)
    return train_ds, val_ds, adapt_ds        

    
def launch_training(seeds, optimizers, datasets, wl):
    for dataset in datasets:
        tf.keras.backend.clear_session()
        X_raw = get_real_data(dataset, 120_000)
    
        data = X_raw[1::2]
        data = data.reshape(-1, 1)
        del X_raw
        gc.collect()
    
        splits = len(data) // 10
    
        train_data = data[:splits * 9].copy()
        val_data = data[splits * 9 : int(splits * 9.9)].copy()
    
        features = train_data.shape[-1]
        input_shape = (wl, features)            
    
        unique_classes = np.unique(train_data.flatten())
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_data.flatten())
        class_weights_dict = dict(enumerate(class_weights))
        num_classes = len(unique_classes)            
        
        # print_mem(f"Before training model {dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}")
        batch_size = compute_batch_size(len(train_data))
        train_steps = math.ceil(len(train_data) / batch_size)
        val_steps = math.ceil(len(val_data) / batch_size)
    
        train_ds, val_ds, adapt_ds = process_data(train_data, val_data, wl, batch_size)
    
        norm = tf.keras.layers.Normalization()
        norm.adapt(adapt_ds)    

        dim = 16
        dropout = 0.1
        s = 0
        train_base_models(norm, train_ds, val_ds, dataset, arch, optimizers,
                          dim, seeds, dropout, epochs, sub_folder, s, num_classes, class_weights_dict)
    

def main():
    seed=42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)  
        
    # === PARAMETERS ===
    epochs = 10
    sub_folder = 'M23_even'
    arch = 'tsmixer'
    wl = 5
    
    target_set = 'Take5'
    seeds = [42, 276, 6, 28]
    optimizers = ['adamw', 'rmsprop', 'adam', 'nadam', 'adamax']
    datasets = ['Euro','Mega','Powerball','C4L','NYLot',
                'Take5','Thunderball','HotPicks',
                'Quick']
    
    launch_training(seeds, optimizers, datasets, wl)


# ==================================================================================================
# ==================================== Training Meta Learner =======================================
# ==================================================================================================

# # --- Generate 3D data sequences ---
# def generate_dataset(X, data, wl=10, features=1):
#     X_test = np.empty([len(data)-wl, wl, features], dtype=np.int8)
#     y_test = np.empty([len(data)-wl, 1], dtype=np.int8)
#     for i in range(len(data)-wl):
#         X_test[i] = X[i:i+wl]
#         y_test[i] = data[i+wl]    
#     return X_test, y_test


# # --- Evaluation ---
# def evaluate_model(model, X_test, name):
#     pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
#     acc = accuracy_score(y_test, pred)
#     print(f'\nLast 10 results for {model.name}...')
#     print(f'Predictions: {pred[-10:]}')
#     print(f'Real Values: {y_test[-10:].flatten()}')
#     print(f'{name} Accuracy: {acc:.4f}')
#     return pred[-1].flatten() 

    
# # --- Meta Model Generator ---
# def create_meta_learner(n_models, num_classes, dim, optimizer, seed, dropout, num_heads=4):
#     initializer1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=seed)
#     initializer2 = tf.keras.initializers.HeNormal(seed=seed)
#     inputs = layers.Input(shape=(n_models, num_classes))
#     num_heads = 4
#     x = layers.Dense(dim, activation="gelu", kernel_initializer=initializer2)(inputs)
#     x = layers.LayerNormalization()(x)

#     for _ in range(2):
#         # Self-attention over the base model outputs
#         mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim//num_heads, seed=seed, kernel_initializer=initializer1)(x,x)
#         mha = layers.Dropout(dropout, seed=seed)(mha)
#         x = layers.LayerNormalization()(x + mha)
#         # FFN
#         ffn = layers.Dense(dim*4, activation='gelu', kernel_initializer=initializer2)(x)
#         ffn = layers.Dropout(dropout, seed=seed)(ffn)
#         ffn = layers.Dense(dim, kernel_initializer=initializer1)(ffn)
#         ffn = layers.Dropout(dropout, seed=seed)(ffn)
#         x = layers.LayerNormalization()(x + ffn)

#     out = layers.GlobalAveragePooling1D()(x)  # shape: (batch, 64)
#     out = layers.Dense(dim // 2, activation='relu', kernel_initializer=initializer2)(out)
#     output = layers.Dense(num_classes, activation="softmax", kernel_initializer=initializer1)(out)
#     model = models.Model(inputs, output, name=f'Meta_L1_{optimizer}_dim{dim}_seed{seed}')
#     model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
#     return model


# # --- Train Meta Models ---
# def train_meta_learner(arch, num_classes, batch_size,
#                        epochs, datasets, sub_folder, optimizers,
#                        dim, seeds, dropout, X_train, y_train, t_dim=64):
        
#     # p_train = []
#     idx = 0
#     n_models = len(datasets) * len(optimizers) * len(seeds)
#     p_train = np.empty((X_train.shape[0], n_models, num_classes), dtype=np.float32)
#     base_models = load_base_models(sub_folder, datasets, arch, optimizers, dim, seeds)  # Get models using generator
#     for bm in base_models:
#         bm_pred = bm(X_train, training=False)
#         p_train[:, idx, :] = bm_pred
#         idx += 1
#         # loss, acc = bm.evaluate(X_train[-100:], y_train[-100:], verbose=0)
#         # print(f'\n{bm.name}     \t| Loss: {loss:.4f}\t| Accuracy: {acc*100:.2f}%')
#         del bm, bm_pred
        
#     # p_train = np.stack(p_train, axis=1)  # shape: (batch, num_models, num_classes)
#     # p_train = np.concatenate([p_train1, p_train], axis=1) if p_train1 is not None else p_train

#     # num_models = p_train.shape[1]
#     for optimizer in ['adamw']:
#         for seed in [42]:
#             tf.keras.backend.clear_session()
#             callback = [
#                 callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
#                 callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
#             ]
#             model = create_meta_learner(n_models, num_classes, t_dim, optimizer, seed, dropout)
#             model.summary()
#             print(f'\nTraining Meta_L1_{optimizer}_dim{t_dim}_seed{seed}...')
#             model.fit(
#                 p_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.1,
#                       verbose=1, callbacks=callback, class_weight=class_weights_dict
#             )
#             model.save(
#                 f'test_models/{sub_folder}/Meta_L1_{optimizer}_dim{t_dim}_seed{seed}.keras'
#             )                
#             del model
    

# X_raw = get_real_data(target_set, 120_000)
# X_raw = X_raw[1::2]
# X_raw = X_raw.reshape(-1,1)
# features = X_raw.shape[-1]

# unique_classes = np.unique(X_raw.flatten())
# class_weights = compute_class_weight('balanced', classes=unique_classes, y=X_raw.flatten())
# class_weights_dict = dict(enumerate(class_weights))
# num_classes = len(unique_classes) 

# X_test, y_test = generate_dataset(X_raw, X_raw, wl, features)
# batch_size = compute_batch_size(len(X_test))
# print(f'\nLength of training data: {len(X_test)}\n')

# meta1, p_train1 = None, None
# # The Meta learner trains on base models
# tf.keras.backend.clear_session()
# t_dim = 256
# train_meta_learner(arch, num_classes, batch_size,
#                    epochs, datasets, sub_folder, optimizers,
#                    dim, seeds, dropout, X_test, y_test, t_dim=t_dim)


# ==================================================================================================
# ========================================= Inference ==============================================
# ==================================================================================================

# def get_base_preds(pred_data_reshaped, sub_folder, datasets, arch, optimizers, dim, seeds):
#     # Compute total number of models
#     n_models = len(datasets) * len(optimizers) * len(seeds)
#     batch_size = pred_data_reshaped.shape[0]
#     num_classes = 10
    
#     p_train = np.empty((batch_size, n_models, num_classes), dtype=np.float32)
    
#     idx = 0
#     s = 0
#     for dataset in datasets:
#         for optimizer in optimizers:
#             for seed in seeds:
#                 model_path = f'test_models/{sub_folder}/{dataset}_{arch}_{optimizer}_dim{dim}_seed{seed}_s{s}.keras'
#                 model = saving.load_model(model_path, compile=False)
                
#                 # bm_pred = model.predict(pred_data_reshaped, verbose=0)
#                 bm_pred = model(pred_data_reshaped, training=False)
#                 p_train[:, idx, :] = bm_pred
                
#                 # Structured summary
#                 bm_pred = np.argmax(bm_pred, axis=1).flatten()
#                 print(f"[{idx}] {model.name}     : {bm_pred}")
                
#                 del bm_pred, model
#                 tf.keras.backend.clear_session()
#                 idx += 1
#     return p_train
    

# X_raw = get_real_data(target_set, 120_000)
# X_raw = X_raw[1::2]
# X_raw = X_raw.reshape(-1,1)
# features = X_raw.shape[-1]

# fin_res = []
# for _ in range(5):
#     print('\n\n\t\tStarting test inference using preds as updates to data_raw...\n\n')

#     pred_data = X_raw[-wl:]    

#     print(f'\nLast {wl} entries of data to be used for inference: {pred_data.flatten()}')
    
#     # Reshape to 3D format as per model expectation
#     pred_data_reshaped = pred_data.reshape(-1, wl, features)
#     # del pred_data
    
#     tf.keras.backend.clear_session()
#     # tf.keras.config.enable_unsafe_deserialization()
#     print(f'\nGetting probabilities from all models in {sub_folder} for target_set {target_set}...')
#     p_train = get_base_preds(pred_data_reshaped, sub_folder, datasets, arch, optimizers, dim, seeds)
#     del pred_data_reshaped            
    
#     tf.keras.backend.clear_session()
#     meta1 = saving.load_model(f'test_models/{sub_folder}/Meta_L1_adamw_dim{t_dim}_seed42.keras')
#     new_pred = np.argmax(meta1.predict(p_train, verbose=0), axis=1)
#     fin_res.extend(new_pred)
#     del p_train, meta1
    
#     X_raw = np.vstack([X_raw, new_pred], dtype=np.int8)
#     print(f'\nLast {wl} entries of updated data: {X_raw[-wl:].flatten()}')
#     print(f'\n\n\t\tResults so far: {fin_res}\n\n')
#     del new_pred


# ##################################################################################################
if __name__ == "__main__":
    main()