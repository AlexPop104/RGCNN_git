import numpy as np
import matplotlib.pyplot as plt


num_modelnet=40


all_losses=np.load('/home/alex/Alex_documents/RGCNN_git/data/logs/Network_performances/Type_20_02_22_22:47:46_losses.npy' )
plt.plot(all_losses, '-b', label='Training loss')
plt.title('Training loss', fontdict=None, loc='center', pad=None)
plt.show()



test_accuracy=np.load('/home/alex/Alex_documents/RGCNN_git/data/logs/Network_performances/Type_20_02_22_22:47:46test_accuracy.npy')
plt.plot(test_accuracy, '-r', label='Test accuracy')
plt.title('Test accuracy', fontdict=None, loc='center', pad=None)
plt.show()

a=all_losses.shape


nr_epochs=a[0]


confusion_matrix_collection=np.load('/home/alex/Alex_documents/RGCNN_git/data/logs/Network_performances/Type_20_02_22_22:47:46_conf_matrix.npy' )

final_matrix=(confusion_matrix_collection[1+(nr_epochs-1)*num_modelnet:1+(nr_epochs)*num_modelnet])

final_matrix2=np.asarray(final_matrix)

np.savetxt("/home/alex/Alex_documents/RGCNN_git/data/conf_matrix_final.csv", final_matrix2, delimiter=",")







