from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
from os_elm import OS_ELM
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm


def softmax(a):
    c = np.max(a, axis=-1).reshape(-1, 1)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=-1).reshape(-1, 1)
    return exp_a / sum_exp_a

def main():
#Autoencoder
    encoding_dim = 32  

    input_img = Input(shape=(784,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)

    encoder = Model(input_img, encoded)

    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    (x_train, t_train), (x_test, t_test) = mnist.load_data()


    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    autoencoder.fit(x_train, x_train,
                    epochs=10,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    encoded_imgs = encoder.predict(x_train)
    x_train = decoder.predict(encoded_imgs)

    encoded_imgs = encoder.predict(x_test)
    x_test = decoder.predict(encoded_imgs)

    
    

    n = 10  
    plt.figure(figsize=(20, 4))
    for i in range(n):
        
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(x_train[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    print("Output of the Autoencoder")
    plt.show()

#OS-ELM    
    n_input_nodes = 784
    n_hidden_nodes = 1024
    n_output_nodes = 10

    os_elm = OS_ELM(
        n_input_nodes=n_input_nodes,
        n_hidden_nodes=n_hidden_nodes,
        n_output_nodes=n_output_nodes,
        loss='mean_squared_error',
        activation='sigmoid',
    )
    n_classes = n_output_nodes

    t_train = to_categorical(t_train, num_classes=n_classes)
    t_test = to_categorical(t_test, num_classes=n_classes)
    t_train = t_train.astype(np.float32)
    t_test = t_test.astype(np.float32)
  
    border = int(1.5 * n_hidden_nodes)
    x_train_init = x_train[:border]
    x_train_seq = x_train[border:]
    t_train_init = t_train[:border]
    t_train_seq = t_train[border:]

    pbar = tqdm.tqdm(total=len(x_train), desc='initial training phase')
    os_elm.init_train(x_train_init, t_train_init)
    pbar.update(n=len(x_train_init))

    pbar.set_description('sequential training phase')
    batch_size = 64
    for i in range(0, len(x_train_seq), batch_size):
        x_batch = x_train_seq[i:i+batch_size]
        t_batch = t_train_seq[i:i+batch_size]
        os_elm.seq_train(x_batch, t_batch)
        pbar.update(n=len(x_batch))
    pbar.close()
    n = 10
    x = x_test[:n]
    t = t_test[:n]

    y = os_elm.predict(x)

    y = softmax(y)
    

    for i in range(n):
        max_ind = np.argmax(y[i])
        print('========== sample index %d ==========' % i)
        print('estimated answer: class %d' % max_ind)
        print('estimated probability: %.3f' % y[i,max_ind])
        print('true answer: class %d' % np.argmax(t[i]))

    [loss, accuracy] = os_elm.evaluate(x_test, t_test, metrics=['loss', 'accuracy'])
    print('val_loss: %f, val_accuracy: %f' % (loss, accuracy))

    print('saving model parameters...')
    os_elm.save('./checkpoint/model.ckpt')


    os_elm.initialize_variables()


    print('restoring model parameters...')
    os_elm.restore('./checkpoint/model.ckpt')

    [loss, accuracy] = os_elm.evaluate(x_test, t_test, metrics=['loss', 'accuracy'])
    print('val_loss: %f, val_accuracy: %f' % (loss, accuracy))

if __name__ == '__main__':
    main()
