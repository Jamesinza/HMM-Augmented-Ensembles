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

tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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
            hs1_pred = hmm_g.predict(hs1).reshape(-1, 1).astype(np.int16)
            a = len(np.unique(hs1_pred))
            # print(f'\na: {a}')
            hs2_pred = hmm_c.predict(hs2).reshape(-1, 1).astype(np.int16)
            b = len(np.unique(hs2_pred))
            # print(f'\nb: {b}')
            base_features = (np.hstack([base_features, hs1_pred, hs2_pred], dtype=np.int16)
                             if base_features is not None
                             else np.hstack([hs1_pred, hs2_pred], dtype=np.int16))
            hs1, hs2 = hs1_pred, hs2_pred  # Update for potential further iterations
            # print(f'\nhs1:\n{hs1}')
            # print(f'hs2:\n{hs2}\n')

        for i in range(1):
            # GMMHMM Feature
            # print('\nNow running GMMHMM stepped...')
            hmm_gmm = GMMHMM(n_components=c, n_mix=1, covariance_type="full", random_state=rng)
            hmm_gmm.fit(hs3)
            hs3_pred = hmm_gmm.predict(hs3).reshape(-1, 1).astype(np.int16)
            c = len(np.unique(hs3_pred))
            # print(f'\nc: {c}')
            base_features = (np.hstack([base_features, hs3_pred], dtype=np.int16)
                             if base_features is not None
                             else hs3_pred)
            hs3 = hs3_pred  # Update for potential further iterations
            # print(f'\nhs3:\n{hs3}')

        for _ in range(1):
            # PoissonHMM Feature
            # print('\nNow running PoissonHMM normal...')
            hmm_pois = PoissonHMM(n_components=d, random_state=rng)
            hmm_pois.fit(hs4)
            hs4_pred = hmm_pois.predict(hs4).reshape(-1, 1).astype(np.int16)
            d = len(np.unique(hs4_pred))
            # print(f'\nc: {c}')
            base_features = (np.hstack([base_features, hs4_pred], dtype=np.int16)
                             if base_features is not None
                             else hs4_pred)
            hs4 = hs4_pred  # Update for potential further iterations
            # print(f'\nhs4:\n{hs4}')

        for _ in range(1):
            # MultinomialHMM Feature
            # print('\nNow running MultinomialHMM normal...')
            hmm_multi = MultinomialHMM(n_components=e, random_state=rng)
            hmm_multi.fit(hs5)
            hs5_pred = hmm_multi.predict(hs5).reshape(-1, 1).astype(np.int16)
            # print(hs5_pred)
            e = len(np.unique(hs5_pred))
            # print(f'\ne: {e}')
            base_features = (np.hstack([base_features, hs5_pred], dtype=np.int16)
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
            hs1_pred = hmm_g.predict(hs1).reshape(-1, 1).astype(np.int16)
            a = len(np.unique(hs1_pred))
            # print(f'\na: {a}')
            hs2_pred = hmm_c.predict(hs2).reshape(-1, 1).astype(np.int16)
            b = len(np.unique(hs2_pred))
            # print(f'\nb: {b}')
            base_features = (np.hstack([base_features, hs1_pred, hs2_pred], dtype=np.int16)
                             if base_features is not None
                             else np.hstack([hs1_pred, hs2_pred], dtype=np.int16))
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
    X_augmented = np.hstack([hmm_features1, hmm_features2], dtype=np.int16)
    # X_augmented = hmm_features1
    return X_augmented

def load_base_models(sub_folder, datasets, optimizers, dims, seeds, l=0):
    # models = []
    for dataset in datasets:
        for optimizer in optimizers:
            for dim in dims:
                for seed in seeds:
                    if l==0:
                        model = saving.load_model(f'test_models/{sub_folder}/{dataset}_{optimizer}_dim{dim}_seed{seed}.keras')                        
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
    print(f"{name} Accuracy: {acc:.4f}")
    return acc

# --- Meta Model Generator ---
def create_meta_learner(num_models, num_classes, dim, optimizer, seed, dropout, l):
    inputs = layers.Input(shape=(num_models, num_classes))
    x = layers.Dense(dim, activation="relu")(inputs)
    x = layers.LayerNormalization()(x)

    for _ in range(2):
        # Self-attention over the base model outputs
        mha = layers.MultiHeadAttention(num_heads=2, key_dim=dim, seed=seed)(x,x)
        mha = layers.Dropout(dropout, seed=seed)(mha)
        x = layers.LayerNormalization()(x + mha)
        # FFN
        ffn = layers.Dense(dim*2, activation='relu')(x)
        ffn = layers.Dropout(dropout, seed=seed)(ffn)
        ffn = layers.Dense(dim)(ffn)
        ffn = layers.Dropout(dropout, seed=seed)(ffn)
        x = layers.LayerNormalization()(x + ffn)

    out = layers.GlobalAveragePooling1D()(x)  # shape: (batch, 64)
    out = layers.Dense(dim // 2, activation='relu')(out)
    output = layers.Dense(num_classes, activation="softmax")(out)
    model = models.Model(inputs, output, name=f'Meta_L{l}_{optimizer}_dim{dim}_seed{seed}')
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
    return model

# --- Train Meta Models ---
def train_meta_learner(num_classes, batch_size,
                      epochs, datasets, sub_folder, optimizers, dims, seeds, dropout,
                       X_train, X_val, X_test, y_train, y_val, base='base', l=0):
    mp_train = []
    mp_val = []
    mp_test = []
        
    p_train = []
    p_val = []
    p_test = []

    if base=='base':
        datasets = ['Meta_L'] if len(datasets)==1 else datasets
        print(f'\nGetting probabilities from all models...')
        base_models = load_base_models(sub_folder, datasets, optimizers, dims, [42], l)  # Get models using generator
        # p_train = np.stack([bm.predict(X_train, verbose=0) for bm in base_models], axis=1)  # shape: (batch, num_models, num_classes)
        
        # base_models = load_base_models(sub_folder, datasets, optimizers, dims, seeds, l)  # Resetting model generator
        # p_val   = np.stack([bm.predict(X_val, verbose=0)   for bm in base_models], axis=1)
        
        # base_models = load_base_models(sub_folder, datasets, optimizers, dims, seeds, l)
        # p_test  = np.stack([bm.predict(X_test, verbose=0)  for bm in base_models], axis=1)  # Resetting model generator
        for bm in base_models:
            print(f'\nGetting probabilities from {bm.name}...')
            p_train.append(bm.predict(X_train, verbose=0))
            p_val.append(bm.predict(X_val, verbose=0))
            p_test.append(bm.predict(X_test, verbose=0))
            
        p_train = np.stack(p_train, axis=1)  # shape: (batch, num_models, num_classes)
        p_val = np.stack(p_val, axis=1)
        p_test = np.stack(p_test, axis=1)
    else:
        p_train = X_train
        p_val = X_val
        p_test = X_test

    l += 1
    num_models = p_train.shape[1]
    for optimizer in optimizers:
        for dim in dims:
            for seed in [42]:
                tf.keras.backend.clear_session()
                callback = [
                    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=2)
                ]
                model = create_meta_learner(num_models, num_classes, dim, optimizer, seed, dropout, l)
                print(f'\nTraining Meta_L{l}_{optimizer}_dim{dim}_seed{seed}...')
                model.fit(
                    p_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(p_val, y_val),
                          verbose=1, callbacks=callback, class_weight=class_weights_dict
                )
                model.save(
                    f'test_models/{sub_folder}/Meta_L{l}_{optimizer}_dim{dim}_seed{seed}.keras'
                )                
                mp_train.append(model.predict(p_train, verbose=0))
                mp_val.append(model.predict(p_val, verbose=0))
                mp_test.append(model.predict(p_test, verbose=0))
                del model
                                                
    train = np.stack(mp_train, axis=1)
    val = np.stack(mp_val, axis=1)
    test = np.stack(mp_test, axis=1)
    del p_train, p_val, p_test
    return train, val, test

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
dataset = 'Take5'
sub_folder = 'M6'
wl = 10
dropout = 0.1
num_heads = 2
seeds = [42, 95, 138, 276, 1, 2, 3, 5, 8, 13, 21, 34,]  # 55, 89, 144,
         # 233,]  # 377, 610, 987, 1597, 2584]  #, 4181, 6765,
         # 10946, 17711, 28657, 46368, 75025, 121393,
         # 196418, 317811]  #, 514229]
dims = [128]
optimizers = ['adam','adamw','nadam','adamax']
# stats = [np.mean, np.std, np.ptp, scipy.stats.skew, scipy.stats.kurtosis]

datasets = ['Take5','Quick']  #,'Mega','Thunderball','Euro','Powerball','C4L','NYLot','HotPicks','Quick']

X_raw = get_real_data(dataset)
data_raw = X_raw.reshape(-1,1)
extra_features = get_extra_features(data_raw, seeds)
data = np.hstack([data_raw, extra_features], dtype=np.int16)
# data = data_raw
# data = np.load('data/Take5_stats_features.npy')

batch_size = compute_batch_size(len(data))
features = data.shape[-1]
dims = [features]
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

joblib.dump(scaler, f'test_models/{sub_folder}/{dataset}_meta_scaler.joblib')
del X_train_2d, X_val_2d, X_test_2d, scaler

# Reshape back to 3D format
X_train = X_train_scaled.reshape(-1, X_train.shape[1], X_train.shape[2])
X_val = X_val_scaled.reshape(-1, X_val.shape[1], X_val.shape[2])
X_test = X_test_scaled.reshape(-1, X_test.shape[1], X_test.shape[2])

del X_train_scaled, X_val_scaled, X_test_scaled

unique_classes = np.unique(X_raw.flatten())
class_weights = compute_class_weight('balanced', classes=unique_classes, y=X_raw.flatten())
class_weights_dict = dict(enumerate(class_weights))

p_train, p_val, p_test = X_train, X_val, X_test
# The 1st Meta learner trains on base models
tf.keras.backend.clear_session()
p_train, p_val, p_test = train_meta_learner(num_classes, batch_size,
                                            epochs, datasets, sub_folder, optimizers, dims,
                                            seeds, dropout, p_train, p_val, p_test,
                                            y_train, y_val, base='base', l=0)
del X_train, X_val, X_test

# datasets = ['Meta_L']
# # From here on Meta Learners runs in loop on Meta outputs
# tf.keras.backend.clear_session()
# for i in range(1,11):
#     p_train, p_val, p_test = train_meta_learner(num_classes, batch_size,
#                                                 epochs, datasets, sub_folder, optimizers, dims,
#                                                 seeds, dropout, p_train, p_val, p_test,
#                                                 y_train, y_val, base='loop', l=i)


    
