from keras.datasets import boston_housing, mnist, cifar10, imdb

(x_train,y_train),(x_test,y_test) = mnist.load_data()
(x_train2,y_train2),(x_test2,y_test2) = boston_housing.load_data()
(x_train3,y_train3),(x_test3,y_test3) = cifar10.load_data()
(x_train4,y_train4),(x_test4,y_test4) = imdb.load_data(num_words=20000)
num_classes = 10

#
# sequence padding
#
from keras.preprocessing import sequence

x_train4 = sequence.pad_sequences(x_train4,maxlen=80)
x_test4 = sequence.pad_sequences(x_test4,maxlen=80)

#
# onehot encoding
#
from keras.utils import to_categorical

Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)
Y_train3 = to_categorical(y_train3, num_classes)
Y_test3 = to_categorical(y_test3, num_classes)

print(y_train)
print(Y_train)