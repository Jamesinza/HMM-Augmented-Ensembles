import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

seed=42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# --- Parameters ---
NUM_CLASSES = 10
INPUT_DIM = 1
NUM_SAMPLES = 10000
NOISE_LEVEL = 0.1
EPOCHS = 1000
BATCH_SIZE = 64
WL = 10

# --- Synthetic Dataset Generation ---
def generate_dataset(n_samples=NUM_SAMPLES, noise=NOISE_LEVEL, wl=10, features=1):
    x_raw = np.random.randint(0, NUM_CLASSES, size=(n_samples,))
    x = x_raw.reshape(-1,1)
    X_test = np.empty([len(x)-wl, wl, features], dtype=np.int16)
    y_test = np.empty([len(x)-wl, 1], dtype=np.int16)
    for i in range(len(x)-wl):
        X_test[i] = x[i:i+wl]
        y_test[i] = x[i+wl]    
    return X_test, y_test

X, y = generate_dataset()
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

unique_classes = np.unique(y.flatten())
class_weights = compute_class_weight('balanced', classes=unique_classes, y=y.flatten())
class_weights_dict = dict(enumerate(class_weights))

# --- Base Model Generator ---
def create_base_model(hidden_units=128, seed=42, wl=10, features=1):
    model = tf.keras.Sequential([
        layers.Input(shape=(wl,features)),
        layers.LSTM(hidden_units, activation='tanh', seed=seed),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Train Base Models ---
def train_base_models(num_models=3, label="A"):
    models = []
    for i in range(num_models):
        callback = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
        ]
        model = create_base_model(hidden_units=32 + i * 16, seed=i)
        print(f"Training BaseModel_{label}{chr(97+i)}...")
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
                  verbose=0, callbacks=callback, class_weight=class_weights_dict)
        models.append(model)
    return models

base_models_1 = train_base_models(label="1")
base_models_2 = train_base_models(label="2")

# --- Meta Learner ---
def create_meta_learner(base_models, name="MetaLearner", seed=42, wl=10, features=1, hidden=64):
    N = len(base_models)
    inputs = tf.keras.Input(shape=(10,1), name=f"{name}_input")
    base_outputs = [model(inputs) for model in base_models]
    g = layers.LSTM(hidden, activation='tanh')(inputs)
    alpha = layers.Dense(N, activation='softmax')(g)
    stacked_preds = layers.Lambda(lambda plist: tf.stack(plist, axis=1))(base_outputs)
    alpha_expanded = layers.Lambda(lambda a: tf.expand_dims(a, axis=-1))(alpha)
    weighted = layers.Multiply()([stacked_preds, alpha_expanded])
    p_meta = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted)
    model = models.Model(inputs=inputs, outputs=p_meta, name=name)
    model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Train Meta Learners ---
meta_learner_1 = create_meta_learner(base_models_1, name="Meta1", seed=4)
meta_learner_1.summary()
meta_learner_2 = create_meta_learner(base_models_2, name="Meta2", seed=5)
meta_learner_2.summary()

print("Training MetaLearner 1...")
callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
]
meta_learner_1.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
                   verbose=1, callbacks=callback, class_weight=class_weights_dict)

print("Training MetaLearner 2...")
callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
]
meta_learner_2.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
                   verbose=1, callbacks=callback, class_weight=class_weights_dict)

# --- Meta-Meta Learner ---
def create_meta_meta_learner(base_models, name="MetaMetaLearner", seed=42, wl=10, features=1, hidden=32):
    N = len(base_models)
    inputs = tf.keras.Input(shape=(10,1), name=f"{name}_input")
    base_outputs = [model(inputs) for model in base_models]
    g = layers.LSTM(hidden, activation='tanh')(inputs)
    alpha = layers.Dense(N, activation='softmax')(g)
    stacked_preds = layers.Lambda(lambda plist: tf.stack(plist, axis=1))(base_outputs)
    alpha_expanded = layers.Lambda(lambda a: tf.expand_dims(a, axis=-1))(alpha)
    weighted = layers.Multiply()([stacked_preds, alpha_expanded])
    p_meta = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted)
    model = models.Model(inputs=inputs, outputs=p_meta, name=name)
    model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

meta_meta_learner = create_meta_meta_learner([meta_learner_1, meta_learner_2], seed=6)
meta_meta_learner.summary()

print("Training Meta-Meta Learner...")
callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3)
]
meta_meta_learner.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
                      verbose=1, callbacks=callback, class_weight=class_weights_dict)

# --- Evaluation ---
def evaluate_model(model, name):
    pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, pred)
    print(f"{name} Accuracy: {acc:.4f}")
    return acc

def evaluate_ensemble(models, name):
    preds = [model.predict(X_test, verbose=0) for model in models]
    avg_pred = np.mean(preds, axis=0)
    acc = accuracy_score(y_test, np.argmax(avg_pred, axis=1))
    print(f'{name} (Averaged) Accuracy: {acc:.4f}')
    return acc

print("\n--- Final Evaluation ---")
subs = ['a', 'b', 'c']
for i, sub in enumerate(subs):
    evaluate_model(base_models_1[i], f"BaseModel_1{sub}")
    evaluate_model(base_models_2[i], f"BaseModel_2{sub}")
evaluate_ensemble(base_models_1 + base_models_2, "All Base Models Ensemble")
evaluate_model(meta_learner_1, "MetaLearner_1")
evaluate_model(meta_learner_2, "MetaLearner_2")
evaluate_model(meta_meta_learner, "MetaMetaLearner")
