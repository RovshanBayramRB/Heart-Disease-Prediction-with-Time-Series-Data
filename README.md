# Heart Disease Prediction with Time Series Data

Binary classification of single-heartbeat ECG recordings as **ischemic** or **normal**, comparing three deep learning architectures — a dense network, a 1D CNN, and a stacked LSTM — on the UCR ECG200 benchmark.

The interesting question here isn't which model wins on accuracy. It's whether an architecture that explicitly models sequence structure beats one that ignores it entirely, on a dataset of 100 training examples.

---

## Dataset

**ECG200** from the [UCR Time Series Classification Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/). Each series traces the electrical activity recorded during a single heartbeat.

| | Train | Test |
|---|---|---|
| Series | 100 | 100 |
| Length | 96 timesteps | 96 timesteps |
| Normal (`1`) | 69 | 64 |
| Ischemia (`-1` → `0`) | 31 | 36 |

No missing values. The class split is roughly 2:1 in both partitions, and the train/test division is the archive's standard split, so results are directly comparable to published benchmarks.

Labels are remapped from `{-1, 1}` to `{0, 1}` and one-hot encoded for a softmax output layer.

**This dataset is small.** 100 training series, of which 10–20 are held back for validation. One test sample is worth a full percentage point of accuracy — a fact worth keeping in mind when reading the results below.

---

## Preprocessing

Inputs are standardized with `StandardScaler`, fitted on training data only and applied to both splits — the correct order, with no leakage from test into training.

Note that this normalizes **per timestep** (each of the 96 positions gets its own mean and standard deviation across series), and that ECG200 arrives from UCR **already z-normalized per series** (each series has mean 0, standard deviation ≈ 0.995). So the scaling being applied is a second, different normalization on top of the archive's own. It aligns the amplitude distribution at each point in the cardiac cycle across patients, which is defensible, but it isn't the usual choice for time series and is worth a deliberate decision rather than a default.

For the CNN and LSTM, a channel dimension is added: `(100, 96)` → `(100, 96, 1)`.

---

## Models

All three use categorical cross-entropy, Adam, batch size 16, up to 100 epochs, with `EarlyStopping` and `ModelCheckpoint`.

### ANN (Multilayer Perceptron)

```
Input(96) → Dense(96, relu) → Dropout(0.1)
          → Dense(48, relu)
          → Dense(24, relu) → Dropout(0.2)
          → Dense(24, relu)
          → Dense(2, softmax)
```

Treats the 96 timesteps as 96 independent features. Temporal ordering is discarded entirely — shuffle the columns and this model performs identically. It's the baseline that answers "does sequence structure actually matter here?"

### CNN (1D Convolutional)

```
Input(96, 1) → Conv1D(128, k=8, same, relu)
             → Conv1D(256, k=5, same, relu)
             → Conv1D(128, k=3, same, relu)
             → GlobalAveragePooling1D
             → Dense(64, relu) → Dropout
             → Dense(32, relu) → Dropout(0.1)
             → Dense(16, relu)
             → Dense(2, softmax)
```

A decreasing kernel schedule (8 → 5 → 3) picks up broad waveform shape first, then finer detail. `GlobalAveragePooling1D` in place of `Flatten` keeps the parameter count down and makes the model translation-invariant along the time axis — useful when the QRS complex isn't perfectly aligned between recordings.

### RNN (Stacked LSTM)

```
Input(96, 1) → LSTM(48, return_sequences) → Dropout(0.2)
             → LSTM(24, return_sequences) → Dropout(0.1)
             → LSTM(12)
             → Dense(2, softmax)
```

Processes the heartbeat sequentially, with the final LSTM collapsing to a single vector for classification. The architecture most explicitly designed for sequences.

---

## Results

On the held-out test set:

| Model | Test loss | Test accuracy | Ischemia recall | Ischemia precision |
|---|---|---|---|---|
| ANN | 0.4936 | **0.88** | 0.75 | 0.90 |
| CNN | **0.4786** | 0.87 | **0.78** | 0.85 |
| LSTM | 0.4912 | 0.77 | 0.67 | 0.69 |

### Reading these numbers honestly

**ANN and CNN are tied.** They differ by one percentage point on a 100-sample test set — literally one sample. Their losses differ by 0.015. Neither gap survives any reasonable notion of statistical significance; picking a winner between them from these numbers is not supportable.

**The LSTM clearly underperforms**, and that gap (11 points) is real. This is the project's actual finding: on 100 training series, the architecture with the most sequential inductive bias does the worst. LSTMs have the most parameters to fit per unit of signal and depend on learning long-range dependencies from data — and 100 examples is not enough to learn them. The convolutional model, which imposes locality as a hard architectural constraint rather than learning it, generalizes better from the same data.

**Ischemia recall is the metric that matters.** Class 0 is the disease class. In a screening context, a missed ischemic beat costs far more than a false alarm, so recall on class 0 is the clinically relevant number — and all three models are weakest exactly there. The best model misses 22% of ischemic beats while overall accuracy reads a comfortable 87%. That gap between headline accuracy and disease-class recall is the standard trap in imbalanced medical classification, and it's why accuracy alone shouldn't be the reported metric here.

By that measure the CNN (0.78 recall) is the better model, not the ANN (0.75) — a conclusion that lines up with the loss column but for a substantive reason rather than a 0.015 difference.

---

## Running it

```bash
git clone https://github.com/RovshanBayramRB/Heart-Disease-Prediction-with-Time-Series-Data.git
cd Heart-Disease-Prediction-with-Time-Series-Data
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
jupyter notebook "Heart Disease Prediction using Time Series Data.ipynb"
```

Written for Google Colab with the data in Google Drive — replace the Drive paths in cell 5 with the local `.tsv` filenames to run elsewhere.

Training is fast on CPU; the dataset is tiny.

---

## Repository structure

```
.
├── Heart Disease Prediction using Time Series Data.ipynb   # ANN, CNN, LSTM
├── ECG200_TRAIN.tsv   # 100 series, 96 timesteps
├── ECG200_TEST.tsv    # 100 series, 96 timesteps
└── README.md
```
