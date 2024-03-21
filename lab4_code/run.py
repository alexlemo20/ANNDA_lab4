from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet

def digit_occurrences(targets, title="Training Set"):
        plt.figure()
        plt.bar(np.unique(targets), [sum(targets==x) for x in np.unique(targets)])
        plt.xticks(np.unique(targets))
        plt.xlabel("Digit")
        plt.title(title)
        plt.savefig(title + 'digit_occurences.png')


if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)


    ''' restricted boltzmann machine '''
    
    # Display a histogram of digit occurences
    train_labels = np.argmax(train_lbls, axis=1)
    test_labels = np.argmax(test_lbls, axis=1)

    digit_occurrences(train_labels)
    digit_occurrences(test_labels, title="Test Set")

    # Display some digit examples
    demo_indices_train_split = []
    for n in range(10):
        demo_indices_train_split.append(np.where(train_labels==n)[0][0])
    fig = plt.figure()
    _, axes = plt.subplots(2, 5, figsize=(10,5))

    for index, ax in zip(demo_indices_train_split, axes.flatten()):
        ax.imshow(train_imgs[index,:].reshape(28, 28), cmap='gray')
        ax.set_title("Number "+str(train_labels[index]))
        ax.axis('off')
    plt.savefig('digit_images.png')


    print ("\nStarting a Restricted Boltzmann Machine..")
    
    # Define mini-batch size
    mini_batch = 20

    # rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
    #                                  ndim_hidden=200, # 200,
    #                                  is_bottom=True,
    #                                  image_size=image_size,
    #                                  is_top=False,
    #                                  n_labels=10,
    #                                  batch_size=mini_batch # 10
    # )
    
    # # epochs in [10, 20]
    # epochs = 10
    # iterations = epochs * mini_batch
    # # Each iteration learns a mini-batch
    # rbm.cd1(visible_trainset=train_imgs, n_iterations=10000) # 10000
    
    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")
    
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "top":2000, "lbl":10}, #  "pen":500,
                        image_size=image_size,
                        n_labels=10,
                        batch_size=20
    )
    
    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)

    dbn.recognize(train_imgs, train_lbls)
    
    dbn.recognize(test_imgs, test_lbls)

    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0,digit] = 1
    #     dbn.generate(digit_1hot, name="rbms")

    # ''' fine-tune wake-sleep training '''

    # dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=100000)

    # dbn.recognize(train_imgs, train_lbls)
    
    # dbn.recognize(test_imgs, test_lbls)
    
    # for digit in range(10):
    #    digit_1hot = np.zeros(shape=(1,10))
    #    digit_1hot[0,digit] = 1
    #    dbn.generate(digit_1hot, name="dbn")
