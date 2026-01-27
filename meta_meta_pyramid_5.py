import gc
import random
import scipy
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
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

def hmm_normal_stack(X_raw, rngs):
    """
    Augment the raw input with HMM-based features.
    The raw data X_raw is assumed to be a 2D array.
    """
    base_features = X_raw.copy()
    for rng in rngs:
        # print(f'\nCreating HMM normal features using seed {rng}...')
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
            base_features = np.hstack([base_features, hs1_pred, hs2_pred], dtype=np.int16)
            hs1, hs2 = hs1_pred, hs2_pred  # Update for potential further iterations
            # print(f'\nhs1:\n{hs1}')
            # print(f'hs2:\n{hs2}\n')

        # for i in range(1):
        #     # GMMHMM Feature
        #     # print('\nNow running GMMHMM stepped...')
        #     hmm_gmm = GMMHMM(n_components=c, n_mix=1, covariance_type="full", random_state=rng)
        #     hmm_gmm.fit(hs3)
        #     hs3_pred = hmm_gmm.predict(hs3).reshape(-1, 1).astype(np.int16)
        #     c = len(np.unique(hs3_pred))
        #     # print(f'\nc: {c}')
        #     base_features = np.hstack([base_features, hs3_pred], dtype=np.int16)
        #     hs3 = hs3_pred  # Update for potential further iterations
        #     # print(f'\nhs3:\n{hs3}')

        # for _ in range(1):
        #     # PoissonHMM Feature
        #     # print('\nNow running PoissonHMM normal...')
        #     hmm_pois = PoissonHMM(n_components=d, random_state=rng)
        #     hmm_pois.fit(hs4)
        #     hs4_pred = hmm_pois.predict(hs4).reshape(-1, 1).astype(np.int16)
        #     d = len(np.unique(hs4_pred))
        #     # print(f'\nc: {c}')
        #     base_features = np.hstack([base_features, hs4_pred], dtype=np.int16)
        #     hs4 = hs4_pred  # Update for potential further iterations
        #     # print(f'\nhs4:\n{hs4}')

        # for _ in range(1):
        #     # MultinomialHMM Feature
        #     # print('\nNow running MultinomialHMM normal...')
        #     hmm_multi = MultinomialHMM(n_components=e, random_state=rng)
        #     hmm_multi.fit(hs5)
        #     hs5_pred = hmm_multi.predict(hs5).reshape(-1, 1).astype(np.int16)
        #     # print(hs5_pred)
        #     e = len(np.unique(hs5_pred))
        #     # print(f'\ne: {e}')
        #     base_features = np.hstack([base_features, hs5_pred], dtype=np.int16)
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
    base_features = X_raw.copy()
    for rng in rngs:
        # print(f'\nCreating HMM stepped features using seed {rng}...')
        # Make copies for HMM-based features
        hs1 = X_raw.copy()
        hs2 = X_raw.copy()
        hs3 = X_raw.copy()

        # --- HMM Feature Augmentation ---
        # Loop over a single iteration or more if needed.
        a=b=c=9
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
            base_features = np.hstack([base_features, hs1_pred, hs2_pred], dtype=np.int16)
            hs1, hs2 = hs1_pred, hs2_pred  # Update for potential further iterations
            # print(f'\nhs1:\n{hs1}')
            # print(f'hs2:\n{hs2}\n')
    return base_features

def get_extra_features(a, stats, wl, rng):
    stacked_data = a[200:]
    print('\nGenerating HMM features...')
    for hmm in [hmm_normal_stack]:  #, hmm_stepped_stack]:
        array = np.empty([len(a)-200, 23], dtype=np.float32)
        for i in range(len(a)-200):
            res = hmm(a[i:i+200], rng)
            array[i] = res[-1]
        # Stack all new features.
        stacked_data = np.hstack([stacked_data, array], dtype=np.float32)
    print('\nGenerating rolling features...')
    data = rolling_stats(X_raw, stacked_data, stats, wl)
    return data    

def rolling_stats(a, stacked_data, stats, wl):
    data = a[wl:]
    for stat in stats:
        array = np.empty([len(a)-wl, 1], dtype=np.float32)
        for i in range(len(a)-wl):
            array[i] = stat(a[i:i+wl])
        # Stack all new features.
        stacked_data = np.hstack([stacked_data, array[200-wl:]])
    return stacked_data    

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
        ffn = layers.Dense(dim*4, activation='relu')(x)
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
def train_meta_learner(base_models, num_classes, batch_size,
                      epochs, sub_folder, optimizers, dims, seeds, dropout,
                       X_train, X_val, X_test, y_train, y_val, base='base', l=0):
    models = []
    mp_train = []
    mp_val = []
    mp_test = []
    print(f'\nGetting probabilities from {len(base_models)} {base} models...')
    p_train = np.stack([bm.predict(X_train, verbose=0) for bm in base_models], axis=1)  # shape: (N, K, C)
    p_val   = np.stack([bm.predict(X_val, verbose=0)   for bm in base_models], axis=1)
    p_test  = np.stack([bm.predict(X_test, verbose=0)  for bm in base_models], axis=1)    
    for optimizer in optimizers:
        for dim in dims:
            for seed in seeds:
                tf.keras.backend.clear_session()
                gc.collect()
                callback = [
                    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
                ]
                model = create_meta_learner(len(base_models), num_classes, dim, optimizer, seed, dropout, l)
                print(f'\nTraining Meta_L{l}_{optimizer}_dim{dim}_seed{seed}...')
                model.fit(p_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(p_val, y_val),
                          verbose=1, callbacks=callback, class_weight=class_weights_dict)
                model.save(f'test_models/{sub_folder}/Meta_L{l}_{optimizer}_dim{dim}_seed{seed}.keras')
                models.append(model)
                
                mp_train.append(model.predict(p_train, verbose=0))
                mp_val.append(model.predict(p_val, verbose=0))
                mp_test.append(model.predict(p_test, verbose=0))
    train = np.stack(mp_train, axis=1)
    val = np.stack(mp_val, axis=1)
    test = np.stack(mp_test, axis=1)
    del p_train, p_val, p_test
    return models, train, val, test

# --- Base Model Generator ---
def create_base_model(optimizer, dim, seed, dropout, num_classes):
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

    model = models.Model(inputs, out, name=f'BaseModel_{optimizer}_{dim}_{seed}')
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return model

# --- Train Base Models ---
def train_base_models(optimizers, dims, seeds, dropout, num_classes, batch_size,
                      epochs, sub_folder, X_train, X_val, y_train, y_val):
    models = []
    for optimizer in optimizers:
        for dim in dims:
            for seed in seeds:
                tf.keras.backend.clear_session()
                gc.collect()        
                callback = [
                    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
                ]
                model = create_base_model(optimizer, dim, seed, dropout, num_classes)
                print(f'\nTraining BaseModel_{optimizer}_dim{dim}_seed{seed}...')
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                          verbose=1, callbacks=callback, class_weight=class_weights_dict)
                model.save(f'test_models/{sub_folder}/BaseModel_{optimizer}_dim{dim}_seed{seed}.keras')
                models.append(model)
    return models

# --- Get real world data ---
def get_real_data(dataset):
    print('\nBuilding dataframe using real data...')

    if dataset == 'Euro' or dataset == 'Thunderball' or dataset == 'Take5':
        cols = ['A', 'B', 'C', 'D', 'E']
    else:
        cols = ['A', 'B', 'C', 'D', 'E', 'F']

    if dataset == 'Take5':
        df = pd.read_csv(f'datasets/training/{dataset}_Full.csv').astype(np.int8)
    else:
        df = pd.read_csv(f'datasets/UK/{dataset}_ascend.csv').astype(np.int8)

    df = df[cols].dropna().astype(np.int8)
    # Format each element as 2-digit string and then flatten digits into an array.
    df = df.map(lambda x: f'{x:02d}')
    flattened = df.values.flatten()
    full_data = np.array([int(d) for num in flattened for d in str(num)], dtype=np.int8)
    return full_data

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
batch_size = 512
dataset = 'Take5'
sub_folder = 'M5'
wl = 10
seeds = [42,95,138,276]  #[1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,
         # 233, 377, 610, 987, 1597, 2584, 4181, 6765,
         # 10946, 17711, 28657, 46368, 75025, 121393,
         # 196418, 317811, 514229, 42, 95, 138, 276]
dims = [64,128,256,512]
dropout = 0.1
num_heads = 2
optimizers = ['adam','adamw','nadam','adamax','rmsprop']
stats = [np.mean, np.std, np.ptp, scipy.stats.skew, scipy.stats.kurtosis]

X_raw = get_real_data(dataset)
data_raw = X_raw.reshape(-1,1)
data = data_raw
# data = np.load('data/Take5_stats_features.npy')

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

# Reshape back to 3D format
X_train = X_train_scaled.reshape(-1, X_train.shape[1], X_train.shape[2])
X_val = X_val_scaled.reshape(-1, X_val.shape[1], X_val.shape[2])
X_test = X_test_scaled.reshape(-1, X_test.shape[1], X_test.shape[2])

joblib.dump(scaler, f'test_models/scaler.joblib')

unique_classes = np.unique(X_raw.flatten())
class_weights = compute_class_weight('balanced', classes=unique_classes, y=X_raw.flatten())
class_weights_dict = dict(enumerate(class_weights))

base_models = train_base_models(optimizers, dims, seeds, dropout, num_classes, batch_size,
                                epochs, sub_folder, X_train, X_val, y_train, y_val)

p_train1, p_val1, p_test1 = X_train.copy(), X_val.copy(), X_test.copy()
for i in range(3):
    base_models, p_train1, p_val1, p_test1 = train_meta_learner(base_models, num_classes, batch_size,
                                                           epochs, sub_folder, optimizers, dims,
                                                           seeds, dropout, X_train, X_val, X_test,
                                                           y_train, y_val, base='base', l=i)

# meta_2, p_train2, p_val2, p_test2 = train_meta_learner(meta_1, num_classes, batch_size,
#                                                        epochs, sub_folder, optimizers,
#                                                        dims=[64], seeds=seeds, dropout=dropout,
#                                                        X_train=p_train1, X_val=p_val1, X_test=p_test1,
#                                                        y_train=y_train, y_val=y_val, base='meta_1', l=2)

# meta_3, p_train3, p_val3, p_test3 = train_meta_learner(meta_2, num_classes, batch_size, epochs=100,
#                                                        sub_folder=sub_folder, optimizers=['adamw'],
#                                                        dims=[128], seeds=[42], dropout=dropout,
#                                                        X_train=p_train2, X_val=p_val2, X_test=p_test2,
#                                                        y_train=y_train, y_val=y_val, base='meta_2', l=3)

# # Evaluate Base Models
# print('\n')
# for i, model in enumerate(base_models):
#     evaluate_model(model, X_test, f'{model.name}')

# # Evaluate Meta Learners    
# print('\n')
# for model in meta_1:
#     evaluate_model(model, p_test1, f'{model.name}')
# print('\n')
# for model in meta_2:
#     evaluate_model(model, p_test2, f'{model.name}')
# print('\n')
# for model in meta_3:
#     evaluate_model(model, p_test3, f'{model.name}')
