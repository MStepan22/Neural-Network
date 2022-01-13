# Import knihoven
import numpy as np
import csv
import random
import math
import matplotlib.pyplot as pltS

#np.random.seed(100)

"""
    PŘÍPRAVA DAT //////////////////////////////////////
"""
with open('cruzeirodosul2010daily.csv', 'r') as f:
    reader = csv.DictReader(f, delimiter=';')
    avg_temps = np.array([l['AvgTemp'] for l in reader])
    avg_temps = np.array([np.nan if l=='' else l for l in avg_temps], dtype='float64') # replace missing val
with open('cruzeirodosul2010daily.csv', 'r') as f:
    reader = csv.DictReader(f, delimiter=';')
    dates = np.array([l['Date'] for l in reader], dtype='int64') / 100_000 # squish the range of dates
with open('cruzeirodosul2010daily.csv', 'r') as f:
    reader = csv.DictReader(f, delimiter=';')
    inputs = np.array([[l['Precipitation'], l['Insolation'], l['Evaporation'], l['AvgHumidity'], l['WindSpeed']] for l in reader])
    inputs = np.where(inputs == '', np.nan, inputs)
    inputs = np.where(inputs == '#N/A', np.nan, inputs)
    inputs = inputs.astype('float64')

# Tisk původních hodnot ze souboru
#print(dates[0:5], avg_temps[0:5], inputs)

"""
# Vygenerování grafu zkoumané časové řady
plt.title("Data před úpravou")
plt.plot(dates, avg_temps, color = 'red')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
"""

@staticmethod
def ratio_list_split(l_split: list, ratio: float) -> tuple:
# def ratio_list_split(l_split: list, ratio: float) -> tuple:
    """
    splits the dataset for training and evaluation
    """
    if not 0 <= ratio <= 1:
        raise ValueError('must be 0 <= ratio <= 1 is:', ratio)

    return (l_split[:int(len(l_split) * ratio)], l_split[int(len(l_split) * ratio):])

@staticmethod
def normalize(z: np.ndarray) -> np.ndarray:
    return z * 1.0/np.nanmax(z)

def linear_interpolation(series: np.ndarray) -> np.ndarray:
    nans, X = _nan_helper(series)
    series[nans] = np.interp(X(nans), X(~nans), series[~nans])
    return series

@staticmethod
def _nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

# Lineární interpolace chybějících dat výstupů, vypočtení popisné statistiky
# střední hodnota a rozptyl (pro 0 stupňů volnosti)
print("------------Střední hodnota a rozptyl------------")
avg_temps_interp = linear_interpolation(avg_temps)
print("střední hodnota | směrodatná odchylka | (před diferenciací a normalizací)")
print(np.mean(avg_temps_interp), np.var(avg_temps_interp))

#print(len(inputs), len(avg_temps_interp))

# Aplikování normalizace, detrendování diferencí I. řádu. Chybějící data doplněna lineární interpolací
norm_avg_temps = normalize(avg_temps_interp)
diff_norm_avg_temps = np.diff(norm_avg_temps)
print("střední hodnota | směrodatná odchylka | (po diferenciaci a normalizaci)")
print(np.mean(diff_norm_avg_temps), np.var(diff_norm_avg_temps))
print("-------------------------------------------------")

inputs_interp = linear_interpolation(inputs)
norm_inputs = normalize(inputs_interp)
diff_norm_inputs = np.diff(norm_inputs, axis=0)

#print(norm_inputs, diff_norm_inputs)


"""
# Vygenerování grafu diferencovaných a normalizovaných dat
plt.title("Diferencovaná a normalizovaná data")
plt.plot(dates[1:], diff_norm_avg_temps)
plt.xlabel("X")
plt.ylabel("Y")
#plt.legend()
plt.show()
"""

# Množina byla rozdělena na trénovací a validační v poměru 75%:25%
# I. varianta
X = np.array(diff_norm_inputs)
y = np.array(diff_norm_avg_temps)

print(X)
print(y)

X_train, X_tests = ratio_list_split(X, 0.75)
y_train, y_tests = ratio_list_split(X, 0.75)
print(len(X_train), len(X_tests), len(y_train), len(y_tests))

# II. varianta
#dataset = [(X, np.array([y])) for X, y in zip(diff_norm_inputs, diff_norm_avg_temps)]
#random.shuffle(dataset)
#print(dataset[0:5])

#training_data, test_data = ratio_list_split(dataset, 0.75)
#print(len(training_data), len(test_data), len(dataset), training_data[0])

"""
    TVORBA SÍTĚ ///////////////////////////////
"""
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

        # funkce tanh
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

        #print(X)

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


"""
    INICIALIZACE SÍTĚ ////////////////////////
"""
if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.pridat_vrstvu(Vrstva(1, 3, 'tanh'))         #skrytá vrstva
    nn.pridat_vrstvu(Vrstva(3, 1, 'tanh'))      #výstupní vrstva

    # Definice vstupů a výstupů
        # Výroková logika OR
            #X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            #y = np.array([[0], [1], [1], [1]])

        # Trénování funkce tanh
            #X = np.array([[-3.0], [-2.5], [-2.0],
            #             [-1.5], [-1.0], [-0.5],
            #              [0.0], [0.5], [1.0],
            #              [1.5], [2.0], [2.5], [3.0]])

            #y = [np.tanh(n) for n in X]


    # Trénování neuronové sítě
    #chyby = nn.trenovani(X, y, 0.5, 201)
    #    print("Přesnost: %.2f%%" % (nn.presnost(nn.predikce(X)[:,0].T, y.flatten()) * 100))
    #print("Očekávané výstupy: \n" + str(y))
    #print("Odhadované výstupy: \n" + str(nn.predikce(X)))


    #outputs = [nn.feed_forward(a) for a in X]
    #print(outputs)


"""
    # Vygenerování grafu porovnání funkce TANH a natrénovanou fcí pomocí NS
    plt.title("Porovnání tanh a natrénované fce")
    plt.plot(X, outputs, color = 'blue')
    plt.plot(X, y, color = 'red')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
"""
"""
    # Vygenerování grafu vývoje chyby MSE
    plt.title('Vývoj chyby MSE')
    plt.plot(chyby, c = 'b', label = 'MSE')
    plt.xlabel('Epochy')
    plt.ylabel('MSE')
    plt.grid(linestyle='-.', linewidth=0.5)
    plt.legend()
    plt.show()
"""