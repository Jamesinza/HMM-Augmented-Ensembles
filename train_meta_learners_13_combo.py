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
        print(f'\nCreating HMM normal features using seed {rng} in {sub_folder} sub_folder...')
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
        print(f'\nCreating HMM stepped features using seed {rng} in {sub_folder} sub_folder...')
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
    # return models

def compute_batch_size(dataset_length):
    base_unit  = 25000
    base_batch = 32
    batch_size = base_batch * math.ceil(dataset_length / base_unit)
    return batch_size                     

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
        if tot_subs == len(sub_folders):
            p_train = np.concatenate([p_train1, p_train], axis=1) if p_train1 is not None else p_train
            p_val   = np.concatenate([p_val1, p_val], axis=1)     if p_train1 is not None else p_val
            p_test  = np.concatenate([p_test1, p_test], axis=1)   if p_train1 is not None else p_test
            
            train_model = True if tot_subs==len(sub_folders) else False
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
                    p_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(p_val, y_val),
                          verbose=1, callbacks=callback, class_weight=class_weights_dict
                )
                model.save(
                    f'test_models/{combo_folder}/cnn//Meta_L{l}_{optimizer}_dim{dim}_seed{seed}.keras'
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
epochs = 100
target_dataset = 'Take5'
sub_folders = ['M13']  #,'M10','M11']
combo_folder = 'combo'
archs = ['cnn','gru', 'lstm']
wl = 10
dropout = 0.7
num_heads = 2
seeds =  [[11,5,16,13,25,3,15,1,14],[4,19,6,12,23,20,8,9,18],[37,38,82,83,46,33,49,410,1000]]
dim = 128
optimizers = ['adamw','adam','nadam','adamax']
datasets = ['Take5'] #,'Quick','Mega','Thunderball','Euro','Powerball','C4L','NYLot','HotPicks']

X_raw = get_real_data(target_dataset, 120_000)
data_raw = X_raw.reshape(-1,1)

data_raw = np.vstack([data_raw[1:], [1]], dtype=np.int8)

# for _ in range(10):
tot_subs = 1
meta1, p_train1, p_val1, p_test1 = None, None, None, None
for i, sub_folder in enumerate(sub_folders):
    extra_features = get_extra_features(data_raw, seeds[i])
    data = np.hstack([data_raw, extra_features], dtype=np.int8)
    
    batch_size = compute_batch_size(len(data))
    features = data.shape[-1]
    dim = features*4
    input_shape = (wl, features)
    print(f'\ninput_shape: {input_shape}\n')
    
    X, y = generate_dataset(data, wl, features)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    
    # Scaling data
    X_train_2d = X_train.reshape(-1, X_train.shape[1]*X_train.shape[2])
    X_val_2d   = X_val.reshape(-1, X_val.shape[1]*X_val.shape[2])
    X_test_2d  = X_test.reshape(-1, X_test.shape[1]*X_test.shape[2])

    tot_datasets = 0
    for dataset in datasets:
        scaler = joblib.load(f'test_models/{sub_folder}/{dataset}_base_scaler.joblib') #MinMaxScaler()
        print(f'\nLoaded Scaler: {dataset}_base_scaler in {sub_folder}')
        
        X_train_scaled = scaler.transform(X_train_2d)
        X_val_scaled   = scaler.transform(X_val_2d)
        X_test_scaled  = scaler.transform(X_test_2d)
        
        # Reshape back to 3D format
        X_train_reshaped = X_train_scaled.reshape(-1, X_train.shape[1], X_train.shape[2])
        X_val_reshaped   = X_val_scaled.reshape(-1, X_val.shape[1], X_val.shape[2])
        X_test_reshaped  = X_test_scaled.reshape(-1, X_test.shape[1], X_test.shape[2])
        
        # del X_train_scaled, X_val_scaled, X_test_scaled
        
        unique_classes     = np.unique(X_raw.flatten())
        class_weights      = compute_class_weight('balanced', classes=unique_classes, y=X_raw.flatten())
        class_weights_dict = dict(enumerate(class_weights))
        
        # The 1st Meta learner trains on base models
        tf.keras.backend.clear_session()
        meta1, p_train1, p_val1, p_test1 = train_meta_learner(
            archs, num_classes, batch_size, epochs, dataset, sub_folder, optimizers, dim,
            [42], dropout, X_train_reshaped, X_val_reshaped, X_test_reshaped,
            y_train, y_val, base='base', l=0, tot_datasets=tot_datasets)
        
    tot_subs += 1

print('\nComparing scores for different Meta Levels')
_ = evaluate_model(meta1, p_test1, 'Meta_L1')

del (p_train1, p_val1, p_test1, X_train, X_val, X_test, X_train_scaled,
     X_val_scaled, X_test_scaled, X_train_reshaped, X_val_reshaped, X_test_reshaped)
# +---------------------------------------------------------+
# +--- Test inference using preds as updates to data_raw ---+
# +---------------------------------------------------------+

'''
    This section of the code as well as optimizations will 
    be sorted later once proof of concept confirmed.
'''
    
updated_data_raw = data_raw  #np.vstack([data_raw, new_pred], dtype=np.int8)
print(f'\nLast 10 entries of updated_data_raw: {updated_data_raw[-10:].flatten()}')

for _ in range(10):

    p_train = []
    for i, sub_folder in enumerate(sub_folders):
        extra_features = get_extra_features(updated_data_raw, seeds[i]) if len(updated_data_raw)>len(data_raw) else extra_features
        data = np.hstack([updated_data_raw, extra_features], dtype=np.int8)         
        
        batch_size = compute_batch_size(len(data))
        features = data.shape[-1]
        dim = features*4
        input_shape = (wl, features)
        print(f'\ninput_shape: {input_shape}\n')
        
        for dataset in datasets:
            scaler = joblib.load(f'test_models/{sub_folder}/{dataset}_base_scaler.joblib') #MinMaxScaler()
            print(f'\nLoaded Scaler: {dataset}_base_scaler in {sub_folder}')

            # Reshape data according saved scaler requirements
            X_train_2d = data[-10:].reshape(-1, wl*features)

            # Scale data
            X_train_scaled = scaler.transform(X_train_2d)
            
            # Reshape to 3D format as per model expectation
            X_train_reshaped = X_train_scaled.reshape(-1, wl, features)

            tf.keras.backend.clear_session()
            print(f'\nGetting probabilities from all models in {sub_folder}...')
            base_models = load_base_models(sub_folder, dataset, archs, optimizers, dim, [42], l=0)  # Get models using generator
    
            for bm in base_models:
                print(f'\nGetting probabilities from {bm.name} in {sub_folder}...')
                p_train.append(bm.predict(X_train_reshaped, verbose=0))
                
    p_train = np.stack(p_train, axis=1)  # shape: (batch, num_models, num_classes)
    
    tf.keras.backend.clear_session()
    new_pred = np.argmax(meta1.predict(p_train, verbose=0), axis=1)
        
    updated_data_raw = np.vstack([updated_data_raw[1:], new_pred], dtype=np.int8)
    print(f'\nLast 10 entries of updated_data_raw: {updated_data_raw[-10:].flatten()}')

   
    