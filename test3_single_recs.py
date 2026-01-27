# Silencing annoying warnings
import shutup
shutup.please()

import gc
import math
import joblib
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from hmmlearn.hmm import GaussianHMM, CategoricalHMM, GMMHMM, PoissonHMM, MultinomialHMM
# from hmmlearn.vhmm import VariationalCategoricalHMM, VariationalGaussianHMM


def create_dataset(sequence, window_length, batch_size, target_pos, shuffle=False):
    """Create a cached and prefetch-enabled tf.data.Dataset."""
    # target_pos = 3
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(window_length + target_pos, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_length + target_pos))
    # Assume the last column is the target; the remaining columns are features.
    ds = ds.map(lambda window: (window[:-target_pos, :-1], window[-1, -1]),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.repeat().prefetch(tf.data.AUTOTUNE)
    return ds


def build_normal_model(input_shape=(10, 1), seed=42, combo='add', dims=256, num_heads=2, dropout=0.5):
    inputs = tf.keras.layers.Input(input_shape)
    x1 = x2 = x3 = inputs

    for _ in range(2):
        x1 = tf.keras.layers.GRU(dims, return_sequences=True, seed=seed)(x1)
        x1 = tf.keras.layers.Dropout(dropout, seed=seed)(x1)

    for _ in range(2):
        x2 = tf.keras.layers.LSTM(dims, return_sequences=True, seed=seed)(x2)
        x2 = tf.keras.layers.Dropout(dropout, seed=seed)(x2)

    if combo == 'add':
        x3 = tf.keras.layers.Add()([x1, x2])
    else:
        x3 = tf.keras.layers.Concatenate()([x1, x2])

    for _ in range(2):
        mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=128, seed=seed)(x3, x3)
        x3 = tf.keras.layers.Dropout(dropout, seed=seed)(mha)

    out = tf.keras.layers.GlobalAveragePooling1D()(x3)
    out = tf.keras.layers.Dense(dims // 2, activation='relu')(out)
    out = tf.keras.layers.Dense(dims // 4, activation='relu')(out)
    out = tf.keras.layers.Dense(10, activation='softmax')(out)
    return tf.keras.Model(inputs, out)


def build_abnormal_model(input_shape=(10, 1), seed=42, dims=256, num_heads=2, dropout=0.5):
    inputs = tf.keras.layers.Input(input_shape)
    x1 = x2 = x3 = inputs
    x3 = tf.keras.layers.Dense(dims, activation='relu')(x3)

    for _ in range(2):
        x1 = tf.keras.layers.GRU(dims, return_sequences=True, seed=seed)(x1)
        x1 = tf.keras.layers.Dropout(dropout, seed=seed)(x1)

        x2 = tf.keras.layers.LSTM(dims, return_sequences=True, seed=seed)(x2)
        x2 = tf.keras.layers.Dropout(dropout, seed=seed)(x2)

        mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=128, seed=seed)(x1, x2, x3)
        x3 = tf.keras.layers.Dropout(dropout, seed=seed)(mha)

    out = tf.keras.layers.GlobalAveragePooling1D()(x3)
    out = tf.keras.layers.Dense(dims // 2, activation='relu')(out)
    out = tf.keras.layers.Dense(dims // 4, activation='relu')(out)
    out = tf.keras.layers.Dense(10, activation='softmax')(out)
    return tf.keras.Model(inputs, out)


def hmm_normal_stack(X_raw, rngs):
    """
    Augment the raw input with HMM-based features.
    The raw data X_raw is assumed to be a 2D array.
    """
    base_features = X_raw.copy()
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
        for _ in range(2):
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

        for i in range(1):
            # GMMHMM Feature
            # print('\nNow running GMMHMM stepped...')
            hmm_gmm = GMMHMM(n_components=c, n_mix=1, covariance_type="full", random_state=rng)
            hmm_gmm.fit(hs3)
            hs3_pred = hmm_gmm.predict(hs3).reshape(-1, 1).astype(np.int16)
            c = len(np.unique(hs3_pred))
            # print(f'\nc: {c}')
            base_features = np.hstack([base_features, hs3_pred], dtype=np.int16)
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
            base_features = np.hstack([base_features, hs4_pred], dtype=np.int16)
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
            base_features = np.hstack([base_features, hs5_pred], dtype=np.int16)
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
    base_features = X_raw.copy()
    for rng in rngs:
        print(f'\nCreating HMM stepped features using seed {rng}...')
        # Make copies for HMM-based features
        hs1 = X_raw.copy()
        hs2 = X_raw.copy()
        hs3 = X_raw.copy()

        # --- HMM Feature Augmentation ---
        # Loop over a single iteration or more if needed.
        a=b=c=9
        for i in range(2):
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


def get_extra_features(X_raw, rng):
    """
    Create HMM-based and NVG-based features.
    """
    hmm_features1 = hmm_normal_stack(X_raw, rng)
    hmm_features2 = hmm_stepped_stack(X_raw, rng)

    # Stack all new features.
    X_augmented = np.hstack([hmm_features1, hmm_features2], dtype=np.int16)
    return X_augmented


def get_real_data(num_samples, dataset):
    """Load and preprocess real data from a CSV."""
    print('\nBuilding dataframe using real data...')

    if dataset == 'Euro' or dataset == 'Thunderball' or dataset == 'Take5':
        cols = ['A', 'B', 'C', 'D', 'E']
    else:
        cols = ['A', 'B', 'C', 'D', 'E', 'F']

    if dataset == 'Take5':
        df = pd.read_csv(f'datasets/training/{dataset}_Full.csv').astype(np.int16)
    else:
        df = pd.read_csv(f'datasets/UK/{dataset}_ascend.csv').astype(np.int16)
    # print(df.head())
    df = df[cols].dropna().astype(np.int16)
    # Format each element as 2-digit string and then flatten digits into an array.
    df = df.map(lambda x: f'{x:02d}')
    flattened = df.values.flatten()
    full_data = np.array([int(d) for num in flattened for d in str(num)], dtype=np.int16)
    return full_data


target_pos = 1
batch_size = 128
window_length = 10
epochs = 1000
dataset = 'Take5'
num_samples = 11_645
sub_folder = 'T3_single'

dropout = 0.5
stop_pat = 15
learn_pat = 3
learn_fac = 0.9

raw_data = get_real_data(num_samples, dataset)
X_raw = raw_data[-20000:].reshape(-1, 1)

seeds1 = [196]
seeds = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,
         233, 377, 610, 987, 1597, 2584, 4181,
         6765, 10946, 17711, 28657, 46368, 75025,
         121393, 196418, 317811, 514229]
combos = ['add']  #, 'conc']
archs = ['normal']  #, 'abnormal']
opts = ['rmsprop', 'adam', 'adamw', 'nadam', 'adamax']
opts_list = [tf.keras.optimizers.RMSprop, tf.keras.optimizers.Adam,
             tf.keras.optimizers.AdamW, tf.keras.optimizers.Nadam,
             tf.keras.optimizers.Adamax]
lr = 1e-3
mod1 = []
mod2 = []
mod3 = []
mod4 = []
mod5 = []
mods = [mod1, mod2, mod3, mod4, mod5]

num_future_steps = 10
future_preds = []

for step in range(num_future_steps):
    # y_data = X_raw.copy()  # len of y_data should match updated X_raw after each iteration
    scaler = joblib.load(f'test_models/{sub_folder}/scaler_{step}.joblib')
    
    # if step == 0:
    data_features = np.load(f'test_models/{sub_folder}/train_data_features_{step}.npy')  # load processed data from file
    # Scale data using original scaler
    val_data_scaled = scaler.transform(data_features)
    # else:
    #     data_features = get_extra_features(X_raw, seeds)
    #     split = 1000
    #     train_data = data_features[:-split]
    #     train_data_y = y_data[:-split]
        
    #     val_data = data_features[-split:]
    #     val_data_y = y_data[-split:] 

    #     # Create new scaler for new training data
    #     scaler = StandardScaler()
    #     train_data_scaled = scaler.fit_transform(train_data)
    #     val_data_scaled = scaler.transform(val_data)

    #     #--- Training data section ---#
    #     # Append unscaled target as final column
    #     train_data = np.hstack([train_data_scaled, train_data_y])  # Align target with augmented features
    #     val_data = np.hstack([val_data_scaled, val_data_y])  # Align target with augmented features
        
    #     train_steps = math.ceil(len(train_data) / batch_size)
    #     val_steps = math.ceil(len(val_data) / batch_size)
        
    #     unique_classes = np.unique(y_data.flatten())
    #     class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_data.flatten())
    #     class_weights_dict = dict(enumerate(class_weights))
        
    #     train_ds = create_dataset(train_data, window_length, batch_size, target_pos, shuffle=True)
    #     val_ds = create_dataset(val_data, window_length, batch_size, target_pos)        
        
    features = data_features.shape[1]
    input_shape = (window_length, features)
    dims = features
    
    test_data = val_data_scaled[-window_length:].copy()
    X_test = test_data.reshape(1, window_length, features)

    all_res = []
    for seed in seeds:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        for arch in archs:
            for i, opt in enumerate(opts):
                if arch == 'normal':
                    for combo in combos:
                        tf.keras.backend.clear_session()
                        gc.collect()
                        # if step == 0:
                        model = tf.keras.models.load_model(f'test_models/{sub_folder}/model_{step}_{opt}_{arch}_{combo}_{seed}.keras')
                        # print(f'\nAdding model_{opt}_{arch}_{combo}_{seed} to list...')
                        res = model.predict(X_test, verbose=0)
                        all_res.append(res)
                        mods[i].extend(np.argmax(res, axis=1))
                        # print(f'\nResults for model_{step}_{opt}_{arch}_{combo}_{seed}: {mods[i]}')
                        # else:
                        #     model = build_normal_model(input_shape, seed, combo, dims, num_heads=2, dropout=dropout)
                        #     optimizer = opts_list[i](learning_rate=1.0) if opt=='adadelta' or opt=='adagrad' else opts_list[i](learning_rate=lr)
                        #     model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                        #                   metrics=['sparse_categorical_accuracy'])
        
                        #     callbacks = [
                        #         tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=stop_pat,
                        #                                          restore_best_weights=True, verbose=1),
                        #         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=learn_fac, patience=learn_pat, cooldown=0)
                        #     ]
                        #     print(f'\nRetraining model_{opt}_{arch}_{combo}_{seed} at step {step}...')
                        #     history = model.fit(train_ds,
                        #                         steps_per_epoch=train_steps,
                        #                         validation_data=val_ds,
                        #                         validation_steps=val_steps,
                        #                         epochs=epochs,
                        #                         callbacks=callbacks,
                        #                         class_weight=class_weights_dict,
                        #                         verbose=0,
                        #                         )                            
                        #     loss, acc = model.evaluate(val_ds, steps=val_steps)
                        #     print(f'\nResults for model_{opt}_{arch}_{combo}_{seed}:\nLoss: {loss}\nAccy: {acc}\n')                            
                        #     res = model.predict(X_test)
                        #     all_res.append(res)
                        #     mods[i].extend(np.argmax(res, axis=1))
                        #     print(f'\nResults for model_{opt}_{arch}_{combo}_{seed}: {mods[i]}')
                else:
                    tf.keras.backend.clear_session()
                    gc.collect()
                    # if step == 0:
                    model = tf.keras.models.load_model(f'test_models/{sub_folder}/model_{step}_{opt}_{arch}_{seed}.keras')
                    res = model.predict(X_test, verbose=0)
                    all_res.append(res)
                    mods[i].extend(np.argmax(res, axis=1))
                    # print(f'\nResults for model_{step}_{opt}_{arch}_{seed}: mods{[i]}')
                    # else:
                    #     model = build_abnormal_model(input_shape, seed, dims, num_heads=2, dropout=dropout)
                    #     optimizer = opts_list[i](learning_rate=1.0) if opt=='adadelta' or opt=='adagrad' else opts_list[i](learning_rate=lr)
                    #     model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                    #                   metrics=['sparse_categorical_accuracy'])
    
                    #     callbacks = [
                    #         tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=stop_pat,
                    #                                          restore_best_weights=True, verbose=1),
                    #         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=learn_fac, patience=learn_pat, cooldown=0)
                    #     ]
                    #     print(f'\nRetraining model_{opt}_{arch}_{seed} at step {step}...')
                    #     history = model.fit(train_ds,
                    #                         steps_per_epoch=train_steps,
                    #                         validation_data=val_ds,
                    #                         validation_steps=val_steps,
                    #                         epochs=epochs,
                    #                         callbacks=callbacks,
                    #                         class_weight=class_weights_dict,
                    #                         verbose=0,
                    #                         )                        
                    #     loss, acc = model.evaluate(val_ds, steps=val_steps)
                    #     print(f'\nResults for model_{step}_{opt}_{arch}_{seed}:\nLoss: {loss}\nAccy: {acc}\n')
                    #     res = model.predict(X_test)
                    #     all_res.append(res)
                    #     mods[i].extend(np.argmax(res, axis=1))
                    #     print(f'\nResults for model_{opt}_{arch}_{seed}: {mods[i]}')

    all_res_array = np.array(all_res)
    # print(f'\nall_res_array:\n{all_res_array}')
    raw_comb_res = np.mean(all_res_array, axis=0)
    fin_comb_res = np.argmax(raw_comb_res, axis=1)
    future_preds.extend(fin_comb_res.flatten())
    print(f'\nStep {step}:\nMean: {fin_comb_res}')

    # Summed version of ensemble probabilities
    summed = np.sum(all_res_array, axis=0)
    ensemble_prob = summed / np.sum(summed, axis=-1, keepdims=True)  # softmax-like normalization
    sum_res = np.argmax(ensemble_prob, axis=1)
    print(f'Sum : {sum_res}')
           
    # # Update raw input sequence with predicted value
    # X_raw = np.vstack([X_raw, [[fin_comb_res[-1]]]], dtype=np.int16)  # Append to raw input
    # print(f'\nLast 3 datapoints of X_raw:\n{X_raw[-3:].flatten()}')
    # print(f'\nResults so far: {future_preds}')
    
# print(f'\nAdam model: {mod1}\t\tRMSprop model: {mod2}')
print(f'\nFinal Reccomendations: {future_preds}')
