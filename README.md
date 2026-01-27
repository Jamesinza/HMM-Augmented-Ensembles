<h1>HMM-Augmented Ensembles</h1>
<blockquote>
A research toolkit that asks a simple, sharp question: <br>
<em>Can simple probabilistic state models (HMMs) manufacture helpful features for neural sequence classifiers?</em><br>
This repository is experimental and not a product.
</blockquote>

<ul>
    <li>Generates HMM-derived discrete features from sliding windows of digit streams.</li>
    <li>Trains ensembles of small RNN base models (GRU + LSTM residuals).</li>
    <li>Trains transformer-style meta-learners that aggregate base model probability outputs.</li>
    <li>Designed to probe representation, calibration, and ensemble effects on noisy/weak-signal sequences.</li>
</ul>

<h2>Repo layout</h2>
<pre>
.
├── datasets/                    # Put CSVs here (Take5_Full.csv, Euro_... etc.)
├── test_models/&lt;sub_folder&gt;/    # Where base/meta checkpoints are saved
├── models/                      # Final model exports
├── main_experiment.py           # Main script (entry point)
├── requirements.txt
└── README.md                    # This file
</pre>

<h2>Requirements</h2>
<p>Tested conceptually against:</p>
<ul>
    <li>Python 3.8+</li>
    <li>TensorFlow 2.x</li>
    <li>numpy, pandas, scipy</li>
    <li>scikit-learn</li>
    <li>hmmlearn</li>
    <li>joblib, matplotlib (for diagnostics)</li>
</ul>
<p>Install quickly:</p>
<pre>
pip install -r requirements.txt
# or individually:
pip install tensorflow numpy pandas scikit-learn hmmlearn scipy joblib matplotlib
</pre>

<h2>Quick start</h2>
<ol>
    <li>Drop minimal CSV(s) into <code>datasets/</code> with expected column formats.</li>
    <li>Edit hyperparameters at the top of <code>main_experiment.py</code> for a quick run:
        <ul>
            <li><code>epochs = 1</code>, <code>seeds = [42]</code>, <code>dims = [64]</code>, <code>wl = 10</code>, <code>batch_size = 128</code></li>
        </ul>
    </li>
    <li>Run:
        <pre>python main_experiment.py</pre>
    </li>
    <li>Check outputs in <code>test_models/&lt;sub_folder&gt;/</code> and the saved scaler <code>test_models/scaler.joblib</code>.</li>
</ol>

<h2>Recommended debug flow</h2>
<ul>
    <li>Set <code>epochs=1</code> and <code>seeds=[42]</code>. Verify pipeline runs end-to-end.</li>
    <li>Disable HMM augmentation and check base model training.</li>
    <li>Re-enable HMMs on a small subset of windows, cache outputs, then run full pipeline using cached features.</li>
    <li>Increase complexity (more seeds, dims, HMM seeds) only after sanity-checking results.</li>
</ul>

<h2>Core concepts</h2>
<ul>
    <li><strong>HMM augmentation:</strong> Fit HMMs on short windows and append predicted discrete states as features.</li>
    <li><strong>Base models:</strong> Diverse small RNNs trained with different optimizers/dimensions/seeds.</li>
    <li><strong>Meta learners:</strong> Transformer-style aggregator that takes base models’ soft probabilities and learns to combine them.</li>
    <li><strong>Diagnostic lens:</strong> Focused on what models learn and how they fail, not raw performance.</li>
</ul>

<h2>Practical tips & gotchas</h2>
<ul>
    <li>HMMs are slow. Precompute and cache features.</li>
    <li>Data types matter. HMM outputs are discrete; cast and scale before combining with floats.</li>
    <li>Use <code>tf.config.experimental.set_memory_growth(gpu, True)</code> to avoid aggressive GPU allocations.</li>
    <li>Expect minor run-to-run variance due to nondeterminism.</li>
    <li>Compute class weights for imbalanced targets.</li>
</ul>

<h2>Diagnostics</h2>
<ul>
    <li>Train/validation loss and accuracy for base models.</li>
    <li>Confusion matrices & class-wise recall.</li>
    <li>Calibration plots comparing base and meta learners.</li>
    <li>Meta-learner attention maps.</li>
    <li>HMM diagnostics: visualize transition matrices and sample sequences.</li>
</ul>

<h2>Suggested experiments</h2>
<ul>
    <li>Ablation: compare raw input, raw + rolling stats, raw + HMM features, all combined.</li>
    <li>Meta effect: sweep number of base models fed to meta learner.</li>
    <li>HMM capacity: vary <code>n_components</code> and observe effect on meta accuracy.</li>
    <li>Caching: precompute HMM stacks and re-run meta experiments quickly.</li>
</ul>

<h2>Expected outcomes</h2>
<ul>
    <li>Meta learner improves: verify whether it exploited genuine signal or artifacts.</li>
    <li>If results are near chance: data likely has negligible predictive structure.</li>
    <li>If base models are overconfident but wrong: investigate overconfidence pathologies.</li>
</ul>

<h2>Troubleshooting</h2>
<ul>
    <li>OOM on GPU: lower <code>batch_size</code> or <code>dim</code>.</li>
    <li>HMM fails to fit: reduce <code>n_components</code> or input size, ensure numeric types.</li>
    <li>Slow HMM runs: parallelize with <code>joblib</code> or subsample windows.</li>
</ul>

<h2>License & credit</h2>
<p>Use freely for research. If publishing results based on these experiments, attribution is appreciated.</p>
