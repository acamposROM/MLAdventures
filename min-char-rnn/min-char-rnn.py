import numpy as np
import sys

# Get filename from arg or default input.txt
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = 'input.txt'

with open(filename, 'r') as f:
    data = f.read()
    
# All unique characters in the data set
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))

char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = {i:ch for i, ch in enumerate(chars)}
print('char_to_ix', char_to_ix)
print('ix_to_char', ix_to_char)

# Hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 16 # number of steps to unroll the RNN for
learning_rate = 1e-1

# Stop after processing this muchdata
MAX_DATA = 1000000

# Model parameters/weights -- these are shared among all steps.
# Weights initialized; randomly biases initialized to 0.
# Inputs are characters one-hot encoded in a vocab sized vector
# Dimensions: H = hidden_size, V = vocab_size
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1))  # output bias

def lossFun(inputs, targets, hprev):
    """Runs forward and backward passes through the RNN.
        inputs, targets: Lists of integers. For some i, inputs[i] is the input
                        character (encoded as an index into the ix_to_char map) and
                        targets[i] is the corresponding next character in the
                        training data (similarly encoded).
        hprev: Hx1 array of initial hidden state
        returns: loss, gradients on model parameters, and last hidden state
    """
    
    xs, hs, ys, ps = {}, {}, {}, {}
    
    # initial incoming state
    hs[-1] = np.copy(hprev)
    loss = 0
    # Forward pass, calculate loss
    for t in range(len(inputs)):
        # Input at time step t is xs[t]. Prepare a one-hot encoded vector of shape
        # (V, 1). inputs[t] is the index where the 1 goes.
        xs[t] = np.zeros((vocab_size, 1)) # encode 1-of-k representation
        xs[t][inputs[t]] = 1
        
        # compute h[t] from h[t-1] and x[t]
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        
        # Compute ps[t] - softmax probabilities for output
        ys[t] = np.dot(Why, hs[t]) + by
        # chatgpt says we can do some optimization here by computing
        # the exponentials once and storing that in a variable.
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        
        # Cross-entropy loss for two probability distributions p and q is defined as
        # follows:
        #
        #   xent(q, p) = -Sum q(k)log(p(k))
        #                  k
        #
        # Where k goes over all the possible values of the random variable p and q
        # are defined for.
        # In our case taking q is the "real answer" which is 1-hot encoded; p is the
        # result of softmax (ps). targets[t] has the only index where q is not 0,
        # so the sum simply becomes log of ps at that index.
        loss += -np.log(ps[t][targets[t], 0])
        
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    
        
def sample(h, seed_ix, n):
    """Sample a sequence of integers from the model.
    Runs the RNN in forward mode for n steps; seed_ix is the seed letter for the
    first time step, and h is the memory state. Returns a sequence of letters
    produced by the model (indices).
    """
    
    # Create a one-hot vector to represent the input
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    
    for t in range(n):
        # Run the foward pass only.
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y)) # softmax
        
        # Sample from the distribution produced by softmax.
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        
        # Prepare input for the next cell
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    
    return ixes
        
# n is the iteration counter; p is the input sequence pointer, at the beginning
# of each step it points at the sequence in the input that will be used for
# training this iteration.
n,p = 0, 0

# Memory variables for Adagrad
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size)*seq_length

while p < MAX_DATA:
    # Prepare inputs
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1)) # reset RNN memory
        p = 0 # go from start of data
    
    # In each step we unroll the RNN for seq_length cells, and present it with
    # seq_length inputs and seq_length target outputs to learn
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
    
    # Sample from the model now and then.
    if n % 1000 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))