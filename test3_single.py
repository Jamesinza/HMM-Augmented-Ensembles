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
# from hmmlearn.hmm import GaussianHMM, CategoricalHMM, GMMHMM, PoissonHMM, MultinomialHMM
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

    df = df[cols].dropna().astype(np.int16)
    # Format each element as 2-digit string and then flatten digits into an array.
    df = df.map(lambda x: f'{x:02d}')
    flattened = df.values.flatten()
    full_data = np.array([int(d) for num in flattened for d in str(num)], dtype=np.int16)
    return full_data


def build_normal_model(input_shape=(10, 1), seed=42, combo='add', dims=256, num_heads=2, dropout=0.5):
    inputs = tf.keras.layers.Input(input_shape)
    x1 = x2 = x3 = inputs

    for _ in range(1):
        x1 = tf.keras.layers.GRU(dims, return_sequences=True, seed=seed)(x1)
        x1 = tf.keras.layers.Dropout(dropout, seed=seed)(x1)

    for _ in range(1):
        x2 = tf.keras.layers.LSTM(dims, return_sequences=True, seed=seed)(x2)
        x2 = tf.keras.layers.Dropout(dropout, seed=seed)(x2)

    if combo == 'add':
        x3 = tf.keras.layers.Add()([x1, x2])
    else:
        x3 = tf.keras.layers.Concatenate()([x1, x2])

    for _ in range(1):
        mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=128, seed=seed)(x3, x3)
        x3 = tf.keras.layers.Dropout(dropout, seed=seed)(mha)

    out = tf.keras.layers.GlobalAveragePooling1D()(x3)
    out = tf.keras.layers.Dense(dims // 2, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout, seed=seed)(out)
    out = tf.keras.layers.Dense(dims // 4, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout, seed=seed)(out)
    out = tf.keras.layers.Dense(10, activation='softmax')(out)
    return tf.keras.Model(inputs, out)


def build_abnormal_model(input_shape=(10, 1), seed=42, dims=256, num_heads=2, dropout=0.5):
    inputs = tf.keras.layers.Input(input_shape)
    x1 = x2 = x3 = inputs
    x3 = tf.keras.layers.Dense(dims, activation='relu')(x3)

    for _ in range(1):
        x1 = tf.keras.layers.GRU(dims, return_sequences=True, seed=seed)(x1)
        x1 = tf.keras.layers.Dropout(dropout, seed=seed)(x1)

        x2 = tf.keras.layers.LSTM(dims, return_sequences=True, seed=seed)(x2)
        x2 = tf.keras.layers.Dropout(dropout, seed=seed)(x2)

        mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=128, seed=seed)(x1, x2, x3)
        x3 = tf.keras.layers.Dropout(dropout, seed=seed)(mha)

    out = tf.keras.layers.GlobalAveragePooling1D()(x3)
    out = tf.keras.layers.Dense(dims // 2, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout, seed=seed)(out)
    out = tf.keras.layers.Dense(dims // 4, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout, seed=seed)(out)
    out = tf.keras.layers.Dense(10, activation='softmax')(out)
    return tf.keras.Model(inputs, out)


target_pos = 1
batch_size = 128
window_length = 10
epochs = 1000
dataset = 'Take5'
sub_folder = 'T3_single'

num_samples = 100_000
num_heads = 2
dropout = 0.5
stop_pat = 15
learn_pat = 3
learn_fac = 0.9
dim_multi = 1

raw_data = get_real_data(num_samples, dataset)
X_raw = raw_data[-20000:].reshape(-1, 1)


seeds = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,
         233, 377, 610, 987, 1597, 2584, 4181,
         6765, 10946, 17711, 28657, 46368, 75025,
         121393, 196418, 317811, 514229]
combos = ['add']
archs = ['normal']
opts = ['rmsprop', 'adam', 'adamw', 'nadam', 'adamax']
opts_list = [tf.keras.optimizers.RMSprop, tf.keras.optimizers.Adam,
             tf.keras.optimizers.AdamW, tf.keras.optimizers.Nadam,
             tf.keras.optimizers.Adamax]
lr = 1e-3

real_res = [0,2,1,2,1,5,2,8,3,1]
X_raw = np.vstack([X_raw, [[0]],[[2]],[[1]],[[2]],[[1]],[[5]],[[2]]], dtype=np.int16)  # Append to raw input

for t in range(7,10):
    # Enhance features using HMM and NVG based approaches
    data = np.load(f'test_models/{sub_folder}/train_data_features_{t}.npy')  # load processed data from file
    features = data.shape[1]
    input_shape = (window_length, features)
    dims = features
    y_data = X_raw.copy()

    split = 1000
    train_data = data[:-split]
    train_data_y = y_data[:-split]
    
    val_data = data[-split:]
    val_data_y = y_data[-split:]
    
    # Scale only the extra features (keep target unscaled)
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    joblib.dump(scaler, f'test_models/{sub_folder}/scaler_{t}.joblib')
    
    # Append unscaled target as final column
    train_data = np.hstack([train_data_scaled, train_data_y])  # Align target with augmented features
    val_data = np.hstack([val_data_scaled, val_data_y])  # Align target with augmented features
    
    train_steps = math.ceil(len(train_data) / batch_size)
    val_steps = math.ceil(len(val_data) / batch_size)
    
    unique_classes = np.unique(train_data_y.flatten())
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_data_y.flatten())
    class_weights_dict = dict(enumerate(class_weights))
    
    train_ds = create_dataset(train_data, window_length, batch_size, target_pos, shuffle=True)
    val_ds = create_dataset(val_data, window_length, batch_size, target_pos)
    
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
                        print(f'\nTraining model_{opt}_{arch}_{combo}_{seed} on train_data_features_{t}...\n')
                        model = build_normal_model(input_shape, seed, combo, dims, num_heads, dropout)
                        # model.summary()
                        optimizer = opts_list[i](learning_rate=lr)
                        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                                      metrics=['sparse_categorical_accuracy'])
    
                        callbacks = [
                            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=stop_pat,
                                                             restore_best_weights=True, verbose=1),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=learn_fac, patience=learn_pat, cooldown=0)
                        ]
    
                        history = model.fit(train_ds,
                                            steps_per_epoch=train_steps,
                                            validation_data=val_ds,
                                            validation_steps=val_steps,
                                            epochs=epochs,
                                            callbacks=callbacks,
                                            class_weight=class_weights_dict,
                                            verbose=0,
                                            )
    
                        model.save(f'test_models/{sub_folder}/model_{t}_{opt}_{arch}_{combo}_{seed}.keras')
                        loss, acc = model.evaluate(val_ds, steps=val_steps, verbose=0)
                        print(f'\nResults for model_{t}_{opt}_{arch}_{combo}_{seed}:\nLoss: {loss}\nAccy: {acc}\n')
    
                else:
                    tf.keras.backend.clear_session()
                    gc.collect()
                    print(f'\nTraining model_{opt}_{arch}_{seed} on train_data_features_{t}...\n')
                    model = build_abnormal_model(input_shape, seed, dims, num_heads, dropout)
                    # model.summary()
                    optimizer = opts_list[i](learning_rate=lr)
                    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                                  metrics=['sparse_categorical_accuracy'])
    
                    callbacks = [
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=stop_pat,
                                                         restore_best_weights=True, verbose=1),
                        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=learn_fac, patience=learn_pat, cooldown=0)
                    ]
    
                    history = model.fit(train_ds,
                                        steps_per_epoch=train_steps,
                                        validation_data=val_ds,
                                        validation_steps=val_steps,
                                        epochs=epochs,
                                        callbacks=callbacks,
                                        class_weight=class_weights_dict,
                                        verbose=0,
                                        )
    
                    model.save(f'test_models/{sub_folder}/model_{t}_{opt}_{arch}_{seed}.keras')
                    loss, acc = model.evaluate(val_ds, steps=val_steps, verbose=0)
                    print(f'\nResults for model_{t}_{opt}_{arch}_{seed}:\nLoss: {loss}\nAccy: {acc}\n')
                    
    X_raw = np.vstack([X_raw, [[real_res[i]]]], dtype=np.int16)  # Append to raw input