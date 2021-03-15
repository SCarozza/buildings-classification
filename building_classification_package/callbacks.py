import matplotlib.pyplot as plt
from IPython.display import clear_output
from keras.callbacks import Callback


class PlotLearning(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.loss = []
        self.val_loss = []
        self.accuracy = []
        self.val_accuracy = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(20, 5))

        clear_output(wait=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.loss, label="loss")
        ax1.plot(self.x, self.val_loss, label="val_loss")
        ax1.legend()

        ax2.plot(self.x, self.accuracy, label="accuracy")
        ax2.plot(self.x, self.val_accuracy, label="validation accuracy")
        ax2.legend()

        plt.show()