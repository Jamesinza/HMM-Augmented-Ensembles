import random
import scipy
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
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

def build_attention_meta_learner(num_models, num_classes):
    """
    Inputs: softmax outputs from K base models: shape=(batch, K, C)
    Applies self-attention across the K base model outputs (not over time),
    and outputs a final softmax prediction.
    """

    inputs = layers.Input(shape=(num_models, num_classes), name="base_probs_input")

    # Optional: project to d_model (can skip this if num_classes is already "enough")
    proj = layers.Dense(64, activation="relu")(inputs)

    # Self-attention over the base model outputs
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=2,
        key_dim=64,  # same as proj dims
        dropout=0.1,
        name="attend_base_models"
    )(proj, proj)

    # Residual + normalization (standard transformer trick)
    x = layers.LayerNormalization()(proj + attn_output)

    # Pool over base models: collapse axis=1
    pooled = layers.GlobalAveragePooling1D()(x)  # shape: (batch, 64)

    # Final dense layer to get class probs
    output = layers.Dense(num_classes, activation="softmax")(pooled)

    model = models.Model(inputs, output, name="AttentionMetaLearner")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )
    return model

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

# --- Meta Learner ---
def create_meta_learner(base_models, name="MetaLearner", optimizer='adam'):
    inputs = tf.keras.Input(shape=input_shape, name=f"{name}_input")
    base_outputs = [model(inputs) for model in base_models]
    x = layers.Concatenate(axis=-1)(base_outputs)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    meta_model = models.Model(inputs=inputs, outputs=out, name=name)
    for model in base_models:
        model.trainable=False
    meta_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return meta_model

def train_meta_learner():
    pass

# --- Base Model Generator ---
def create_base_model(dims=128, optimizer='adam', seed=42):
    inputs = tf.keras.layers.Input(input_shape)
    x1 = x2 = x3 = inputs
    
    for _ in range(2):
        x1 = tf.keras.layers.GRU(dims, return_sequences=True, seed=seed)(x1)
        x1 = tf.keras.layers.Dropout(dropout, seed=seed)(x1)

    for _ in range(2):
        x2 = tf.keras.layers.LSTM(dims, return_sequences=True, seed=seed)(x2)
        x2 = tf.keras.layers.Dropout(dropout, seed=seed)(x2)

    x3 = tf.keras.layers.Add()([x1, x2])

    for _ in range(2):
        mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=128, seed=seed)(x3, x3)
        x3 = tf.keras.layers.Dropout(dropout, seed=seed)(mha)
        
    out = tf.keras.layers.GlobalAveragePooling1D()(x3)
    out = tf.keras.layers.Dense(dims // 2, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout, seed=seed)(out)
    out = tf.keras.layers.Dense(dims // 4, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout, seed=seed)(out)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(out)

    model = tf.keras.Model(inputs, out)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    return model

# --- Train Base Models ---
def train_base_models(optimizers=['adam','adamw','nadam'], seed=42, label="A"):
    models = []
    num_models = len(optimizers)
    for i in range(num_models):
        callback = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
        ]
        optimizer = optimizers[i]
        model = create_base_model(dims=64, optimizer=optimizer, seed=seed)
        print(f"Training BaseModel_{label}{chr(97+i)}...")
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
                  verbose=1, callbacks=callback, class_weight=class_weights_dict)
        model.save(f'test_models/BaseModel_{label}{chr(97+i)}.keras')
        models.append(model)
    return models

# --- Get real world data ---
def get_real_data(dataset):
    """Load and preprocess real data from a CSV."""
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
NUM_CLASSES = 10
NOISE_LEVEL = 0.1
EPOCHS = 1000
BATCH_SIZE = 512
dataset = 'Take5'
wl = 10
seeds = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
dropout = 0.1
num_heads = 2
optimizers = ['adam','adamw','nadam']
stats = [np.mean, np.std, np.ptp, scipy.stats.skew, scipy.stats.kurtosis]

X_raw = get_real_data(dataset)
data_raw = X_raw.reshape(-1,1)
# data = rolling_stats(data_raw, stats, wl)
data = data_raw  #get_extra_features(data_raw, stats, wl, seeds)

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

joblib.dump(scaler, f'test_models//scaler.joblib')

unique_classes = np.unique(X_raw.flatten())
class_weights = compute_class_weight('balanced', classes=unique_classes, y=X_raw.flatten())
class_weights_dict = dict(enumerate(class_weights))

base_models_1 = train_base_models(optimizers, label="1", seed=42)
# base_models_2 = train_base_models(optimizers, label="2", seed=8)
# base_models_3 = train_base_models(optimizers, label="3", seed=5)


# # --- Train Meta Learners ---
# meta_learner_1 = create_meta_learner(base_models_1, name="Meta1")
# meta_learner_2 = create_meta_learner(base_models_2, name="Meta2")
# meta_learner_3 = create_meta_learner(base_models_2, name="Meta3")

# print("Training MetaLearner 1...")
# callback = [
#     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
#     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
# ]
# meta_learner_1.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
#                    verbose=1, callbacks=callback, class_weight=class_weights_dict)
# meta_learner_1.save('test_models/metalearner1.keras')

# print("Training MetaLearner 2...")
# callback = [
#     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
#     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
# ]
# meta_learner_2.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
#                    verbose=1, callbacks=callback, class_weight=class_weights_dict)
# meta_learner_2.save('test_models/metalearner2.keras')

# print("Training MetaLearner 3...")
# callback = [
#     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
#     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
# ]
# meta_learner_3.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
#                    verbose=1, callbacks=callback, class_weight=class_weights_dict)
# meta_learner_3.save('test_models/metalearner3.keras')

# --- Different Experiment ---#
# Step 1: Generate base model predictions
# For training set:
P_train = np.stack([bm.predict(X_train, verbose=0) for bm in base_models_1], axis=1)  # shape: (N, K, C)
P_val   = np.stack([bm.predict(X_val, verbose=0)   for bm in base_models_1], axis=1)
P_test  = np.stack([bm.predict(X_test, verbose=0)  for bm in base_models_1], axis=1)

# Step 2: Train attention-based meta learner
meta_attn = build_attention_meta_learner(num_models=len(base_models_1), num_classes=10)
callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
]
meta_attn.fit(P_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(P_val, y_val),
              verbose=1, callbacks=callback, class_weight=class_weights_dict)

# # Step 3: Evaluate
# print('\n')
# meta_attn.evaluate(P_test, y_test)

# ------------------------------------------------------------------------------------------------------ #


# # --- Train Meta-Meta Learners ---
# meta_meta_learner = create_meta_learner([meta_learner_1, meta_learner_2, meta_learner_3], name='Meta-MetaLearner')

# print("Training Meta-Meta Learner...")
# callback = [
#     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
#     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
# ]
# meta_meta_learner.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
#                       verbose=1, callbacks=callback, class_weight=class_weights_dict)
# meta_meta_learner.save('test_models/meta-metalearner.keras')

print("\n--- Final Evaluation ---")
subs = ['a', 'b', 'c']
for i, sub in enumerate(subs):
    evaluate_model(base_models_1[i], X_test, f"BaseModel_1{sub}")
#     evaluate_model(base_models_2[i], f"BaseModel_2{sub}")
evaluate_model(meta_attn, P_test, "MetaAttention")
# evaluate_model(meta_learner_1, "MetaLearner_1")
# evaluate_model(meta_learner_2, "MetaLearner_2")
# evaluate_model(meta_meta_learner, "MetaMetaLearner")
