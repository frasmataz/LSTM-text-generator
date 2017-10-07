import tensorflow as tf
import numpy
import sys
import boto3
import os.path
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.optimizers import RMSprop

properties = {
            'hidden_dim': 500,
            'seq_length': 500,
            'iterations': 10000,
            'batch_size': 32,
            'gen_length': 500,
            'epochs'    : 4,
            'learning_rate' : 0.025,
            'weights_file': 'weights.w',
            'load_weights': True,
            'load_weights_s3': False
        }

for arg in sys.argv:
    if arg == '-g':
        properties['generate'] = True

def prepare_training_data():
    training_data = open('scp.txt', 'r').read()
    training_data_symbols = list(set(training_data)) 
    int_to_symbol = []
    symbol_to_int = {}
    for idx, val in enumerate(training_data_symbols):
        int_to_symbol.append(val)
        symbol_to_int[val] = idx

    vocab_size = len(symbol_to_int)
    seq_length = properties['seq_length']
    training_data_indexed = []

    for word in training_data:
        training_data_indexed.append(symbol_to_int[word])

    sequences = len(training_data_indexed) / seq_length

    print('Vocab size: {}, Training data is {} symbols long, Sequence length: {}, Sequences: {}'.format(
        vocab_size,
        len(training_data_indexed),
        seq_length,
        sequences
        ))

    X = numpy.zeros((sequences,
                seq_length,
                vocab_size),dtype=float)

    y = numpy.zeros((sequences,
                seq_length,
                vocab_size),dtype=float)
    
    for i in range(0, sequences):
        print('Loading sequence {}/{}'.format(i,sequences))
	X_sequence = training_data_indexed[i * seq_length:(i + 1) * seq_length]
        X_sequence_ix = [value for value in X_sequence]
        input_sequence = numpy.zeros((seq_length, vocab_size))
        for j in range(seq_length):
            input_sequence[j][X_sequence_ix[j]] = 1.
            X[i] = input_sequence

        y_sequence = training_data_indexed[i * seq_length + 1:(i + 1) * seq_length + 1]
        y_sequence_ix = [value for value in y_sequence]
        target_sequence = numpy.zeros((seq_length, vocab_size))
        for j in range(seq_length):
            target_sequence[j][y_sequence_ix[j]] = 1.
            y[i] = target_sequence

    return {
        'data': training_data_indexed,
        'i2s' : int_to_symbol,
        's2i' : symbol_to_int,
        'vocab_size' : vocab_size,
        'X' : X,
        'y' : y
    }

def generate_text(model, length, vocab_size, td):
    # starting with random character
    length = length + 1
    ix = [numpy.random.randint(vocab_size)]
    y_char = [td['i2s'][ix[-1]]]
    X = numpy.zeros((1, length, vocab_size))
    for i in range(length):
        # appending the last predicted character to sequence
        X[0, i, :][ix[-1]] = 1
        sys.stdout.write(td['i2s'][ix[-1]])
        ix = numpy.argmax(model.predict(X[:, :i + 1, :])[0], 1)
        y_char.append(td['i2s'][ix[-1]])
    return ('').join(y_char)

    print('\n')

def load_weights_from_s3():
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket('fsharp-nn-data')
    try:
        my_bucket.download_file(properties['weights_file'], properties['weights_file'])
    except:
        print('Couldn\'t get weights from S3')

def save_weights_to_s3():
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket('fsharp-nn-data')



print('Preparing training data..')
td = prepare_training_data()
print('Finished preparing training data.')

# Set up network
vocab_size = td['vocab_size']

print('Building neural network..')
model = Sequential()
model.add(LSTM(
    properties['hidden_dim'], 
    input_shape=(None, vocab_size),
    return_sequences=True,
    dropout=0.03,
    recurrent_dropout=0.03,
    use_bias=True,
    activation = 'relu'
))
model.add(TimeDistributed(Dense(vocab_size)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=properties['learning_rate'])
print('Compiling network..')
model.compile(loss="categorical_crossentropy", optimizer=optimizer)
print('Done.  Beginning training,')

if 'load_weights_s3' in properties:
    load_weights_from_s3()

if 'load_weights' in properties and os.path.isfile(properties['weights_file']):
    model.load_weights(properties['weights_file'])

if 'generate' in properties:
    for i in range(properties['iterations']):
        generate_text(model, properties['gen_length'], vocab_size, td)
    print()
else:
    generate_text(model, properties['gen_length'], vocab_size, td)

    print()

    for i in range(properties['iterations']):
        model.fit(td['X'], td['y'], properties['batch_size'], verbose=1, epochs=properties['epochs'], shuffle=True)
        model.save_weights(properties['weights_file'])

        generate_text(model, properties['gen_length'], vocab_size, td)
