import matplotlib.pyplot as plt

def plot_tensor(tensor):
    print(tensor.numpy()[0])
    #plt.pcolormesh(tensor.numpy()[0], cmap='hsv')
    plt.imshow(tensor.numpy()[0], cmap='tab10', interpolation='nearest')
    plt.colorbar()
    plt.show()