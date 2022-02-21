from matplotlib import pyplot as plt

PATH = "reports\\figures\\"

def visualize(history, accuracy_or_loss, filename="figure.png"):
    plt.plot(history.history[accuracy_or_loss])
    plt.plot(history.history['val_' + accuracy_or_loss])
    plt.title('model ' + accuracy_or_loss)
    plt.ylabel(accuracy_or_loss)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    plt.savefig(PATH+filename) 
    plt.show()

def visualize_histories(first_history, second_history, accuracy_or_loss="accuracy", 
                        filename ="figure.png", first_name="first", second_name="second", 
                        first_color="red", second_color="blue"):
    plt.plot(first_history.history[accuracy_or_loss], color=first_color)
    plt.plot(first_history.history['val_' + accuracy_or_loss], color=first_color, linestyle='dashed')
    plt.plot(second_history.history[accuracy_or_loss], color=second_color)
    plt.plot(second_history.history['val_' + accuracy_or_loss], color=second_color, linestyle='dashed')
    plt.title('model ' + accuracy_or_loss)
    plt.ylabel(accuracy_or_loss)
    plt.xlabel('epoch')
    plt.legend([first_name+' train', first_name+' validation', second_name+' train', second_name+' validation'], loc='upper left')

    plt.savefig(PATH+filename) 
    plt.show()

def dummy_plot(data):
    _, ax = plt.subplots()
    ax.plot(data["x"], data["y"])
    return ax
