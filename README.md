# Better RNN Components
Tensorflow RNN components implemented as simply as possible

## Contents
@@

## RNN CELLS
A collection of different variants of RNN, LSTM, and GRU cells.

### RNN Cell
Simplest of RNN cells, features a single sigmoid layer at each time step.

### LSTM Cell
Basic LSTM cell as described in [Long Short-Term Memory](http://www.bioinf.jku.at/publications/older/2604.pdf) with optional dropout.

### LSTM Cell with Peepholes
LSTM cell similar to the one above but with added peepholes as described in [Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf).

### mLSTM Cell
Multiplicative variant of LSTM as described in [Multiplicative LSTM For Sequence Modeling](https://arxiv.org/pdf/1609.07959.pdf).

### mLSTM Cell with Peepholes
Multiplicative LSTM with peepholes, combines the concepts of the above two cells.

### mLSTM Cell with L2 regularization
Multiplicative LSTM cell with L2 regularization as described in [L2 Regularization for Learning Kernels](https://arxiv.org/pdf/1205.2653.pdf).

### GRU Cell
GRU cell as described in [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf).

## RNN Presets
Several preset RNNs to use in your code or as a reference.

### Bidirectional LSTM
Encodes inputs in forward and reverse time order and then concatenates the resulting out puts and states using an LSTM cell.

### Bidirectional GRU

### Stacked LSTM

### Stacked GRU

### Luong Attention Mechanism

### Bahdanau Attention Mechanism

### Temporal Attention Mechanism

## Self Critical Loss Function

@desc@

