# Import knihoven
import numpy as np
import math
import matplotlib.pyplot as plt

#np.random.seed(100)

# Tato třída reprezentuje vrstvu (skrytou nebo výstupní) v neuronové síti.
class Vrstva:

    def __init__(self, n_vstupy, n_neurony, aktivacni_fce=None, vahy=None, bias=None):
        """
        Metoda pro inicializaci vrstvy (skryté nebo výstupní).
        :param int n_vstupy: Příslušné vstupy (ze vstupní vrstvy nebo ze skryté vrstvy)
        :param int n_neurony: Počet neuronů ve vrstvě.
        :param str aktivacni_fce: Vybraná aktivační funkce.
        :param vahy: Příslušné váhy vrstvy.
        :param bias: Příslušný bias vrstvy.
        """

        self.vahy = vahy if vahy is not None else np.random.randn(n_vstupy, n_neurony)
        self.aktivacni_fce = aktivacni_fce
        self.bias = bias if bias is not None else np.random.randn(n_neurony)
        self.posledni_aktivace = None
        self.chyba = None
        self.delta = None

    def aktivace(self, x):
        """
        Metoda, která vypočítá r = (X * Wi) + bias.
        :param x: Vstupy.
        :return: Výsledky.
        """

        r = np.dot(x, self.vahy) + self.bias
        self.posledni_aktivace = self._aplikuj_aktivacni_fce(r)

       # print(x, r, self.posledni_aktivace)


        return self.posledni_aktivace



    def _aplikuj_aktivacni_fce(self, r):
        """
        Metoda, která aplikuje vybranou aktivační funkci.
        :param r: Normální hodnota.
        :return: "Aktivovaná" hodnota.
        """

        # Funkce tanh
        if self.aktivacni_fce == 'tanh':
            return np.tanh(r)

        # sigmoid
        if self.aktivacni_fce == 'sigmoid':
            return 1 / (1 + np.exp(-r))

        return r

    def derivace_fce(self, r):
        """
        Metoda, která derivuje aktivační funkci (pro backprop.).
        :param r: Normální hodnota.
        :return: "Derivovaná" hodnota.
        """

        if self.aktivacni_fce == 'tanh':
            return 1 - r ** 2

        if self.aktivacni_fce == 'sigmoid':
            return r * (1 - r)

        return r

# Tato třída reprezentuje neuronovou síť.
class NeuralNetwork:

    def __init__(self):
        self._sit = []

    def pridat_vrstvu(self, vrstva):
        """
        Metoda, která vytvoří vrstvu v neuronové sítě.
        :param Vrstva vrstva: Vrstva k vytvoření.
        """

        self._sit.append(vrstva)

    def feed_forward(self, X):
        """
        Metoda, která zajišťuje dopředný chod sítí (feed forward).
        :param X: Vstupní hodnoty.
        :return: Výsledek.
        """

        for vrstva in self._sit:
            X = vrstva.aktivace(X)

        return X

        print(X)

    def predikce(self, X):
        """
        Metoda, která vrací vítěznou pravděpodobnost a vrací vítězný index.
        :param X: Vstupní hodnoty.
        :return: Predikce.
        """

        ff = self.feed_forward(X)
        return ff

    def backpropagation(self, X, y, learning_rate):
        """
        Metoda, která provádí zpětný chod (backpropagation) a updatuje váhy vrstev.
        :param X: Vstupní hodnoty.
        :param y: Očekávané hodnoty.
        :param float learning_rate: Parametr učení (mezi 0 a 1).
        """

        # Feed forward pro výstupy
        vystup = self.feed_forward(X)

        # Cyklus napříč vrstvami (zpětně)
        for i in reversed(range(len(self._sit))):
            vrstva = self._sit[i]

            # Pokud je vrstva výstupní vrstvou
            if vrstva == self._sit[-1]:
                vrstva.chyba = y - vystup
                # Výstup = vrstva.posledni_aktivace v tomto případě
                vrstva.delta = vrstva.chyba * vrstva.derivace_fce(vystup)
            # Pokud je vrstva skrytou vrstvou
            else:
                dalsi_vrstva = self._sit[i + 1]
                vrstva.chyba = np.dot(dalsi_vrstva.vahy, dalsi_vrstva.delta)
                vrstva.delta = vrstva.chyba * vrstva.derivace_fce(vrstva.posledni_aktivace)

        # Update vah
        for i in range(len(self._sit)):
            vrstva = self._sit[i]
            # Vstup je buď výstup předchozí vrstvy nebo X (vstup) samotný
            vstup_k_pouziti = np.atleast_2d(X if i == 0 else self._sit[i - 1].posledni_aktivace)
            vrstva.vahy += vrstva.delta * vstup_k_pouziti.T * learning_rate

    def trenovani(self, X, y, learning_rate, max_epoch):
        """
        Metoda, která natrénuje neuronovou síť s pomocí backpropagation.
        :param X: Vstupní hodnoty.
        :param y: Očekávané hodnoty.
        :param float learning_rate: Parametr učení (mezi 0 a 1).
        :param int max_epoch: Maximum epoch (cyklů).
        :return: Výpis vypočítaných chyb MSE.
        """

        mses = []

        for i in range(max_epoch):
            for j in range(len(X)):
                self.backpropagation(X[j], y[j], learning_rate)
            if i % 10 == 0:     # každých 10 epoch
                mse = np.mean(np.square(y - nn.feed_forward(X)))
                mses.append(mse)
                print('Epocha: #%s, MSE: %f' % (i, float(mse)))

        return mses

#    @staticmethod
#    def presnost(y_pred, y_true):

#        Metoda, která vypočítá přesnost mezi predikovanými a očekávanými hodnotami.
#        :param y_pred: Predikované hodnoty výstupu.
#        :param y_true: Očekávané hodnoty výstupu.
#        :return: Vypočtená přesnost.

#        return ((np.round(y_pred, 1) == y_true)).mean()

"""
    Inicializace
"""
if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.pridat_vrstvu(Vrstva(1, 10, 'tanh'))         #skrytá vrstva
    nn.pridat_vrstvu(Vrstva(10, 1, 'tanh'))      #výstupní vrstva

    # Definice vstupů a výstupů
    # Výroková logika OR
    #X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    #y = np.array([[0], [1], [1], [1]])

    X = np.array([[-3.0], [-2.5], [-2.0],
                  [-1.5], [-1.0], [-0.5],
                  [0.0], [0.5], [1.0],
                  [1.5], [2.0], [2.5], [3.0]])

    y = [np.tanh(n) for n in X]

    # Trénování neuronové sítě
    chyby = nn.trenovani(X, y, 0.01, 1001)
#    print("Přesnost: %.2f%%" % (nn.presnost(nn.predikce(X)[:,0].T, y.flatten()) * 100))
    print("Očekávané výstupy: \n" + str(y))
    print("Odhadované výstupy: \n" + str(nn.predikce(X)))


    #print(nn._sit)
    #for layer in nn._sit:
    #    print(layer.vahy, layer.bias, layer.delta)

    outputs = [nn.feed_forward(a) for a in X]
    print(outputs)

    plt.title("numpy.tanh()")
    plt.plot(X, outputs, color = 'green')
    plt.plot(X, y, color = 'red')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

"""
    # Vygenerování grafu vývoje chyby MSE
    plt.plot(chyby, c = 'b', label = 'MSE')
    plt.title('Vývoj chyby MSE')
    plt.xlabel('Epochy')
    plt.ylabel('MSE')
    plt.grid(linestyle='-.', linewidth=0.5)
    plt.legend()
    plt.show()
"""

