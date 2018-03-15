import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
sys.path.append('../Models/CNN/')
from model import Inception,VGG
sys.path.append('../')
from training_options import *
from load_data import load_custom
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import precision_score, recall_score
# class options
two = ['Code','No Code']
four = ['VC','PVC','HC','NC']
# training option functions
functions = [code_vs_no_code_strict,code_vs_no_code_partially,code_vs_no_code_partially_handwritten,handwritten_vs_else,handwritten_vs_no_code,all_four]
results = {'code_vs_no_code_strict.h5':{'actual':np.array([]),'predicted':np.array([])},
            'code_vs_no_code_partially.h5':{'actual':np.array([]),'predicted':np.array([])},
            'code_vs_no_code_partially_handwritten.h5':{'actual':np.array([]),'predicted':np.array([])},
            'handwritten_vs_else.h5':{'actual':np.array([]),'predicted':np.array([])},
            'handwritten_vs_no_code.h5':{'actual':np.array([]),'predicted':np.array([])},
            'all_four.h5':{'actual':np.array([]),'predicted':np.array([])}
            }

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
'''
for f in functions:
    print 'new function'
    for fold in range(0,5):
        fold_dir = 'Fold_'+str(fold)+'/'
        # load data from file
        X_TRAIN,Y_TRAIN,X_TEST,Y_TEST = load_data(prefix='../'+fold_dir)
        # get data, model and weights file name from training options function
        x_train,y_train,x_test,y_test,model,weights = f(X_TRAIN,Y_TRAIN,X_TEST,Y_TEST)
        # load weights file
        model.load_weights('../' + fold_dir + weights)
        # predict classes for testing images
        predicitions = model.predict(x_test)

        predicted = np.argmax(predicitions,axis=1)
        actual = np.argmax(y_test,axis=1)

        results[weights]['actual'] = np.append(results[weights]['actual'],actual)
        results[weights]['predicted'] = np.append(results[weights]['predicted'],predicted)

        print fold

np.savez('actual',code_vs_no_code_strict=results['code_vs_no_code_strict.h5']['actual'],
    code_vs_no_code_partially=results['code_vs_no_code_partially.h5']['actual'],
    code_vs_no_code_partially_handwritten=results['code_vs_no_code_partially_handwritten.h5']['actual'],
    handwritten_vs_else=results['handwritten_vs_else.h5']['actual'],
    handwritten_vs_no_code=results['handwritten_vs_no_code.h5']['actual'],
    all_four=results['all_four.h5']['actual'])

np.savez('predicted',code_vs_no_code_strict=results['code_vs_no_code_strict.h5']['predicted'],
    code_vs_no_code_partially=results['code_vs_no_code_partially.h5']['predicted'],
    code_vs_no_code_partially_handwritten=results['code_vs_no_code_partially_handwritten.h5']['predicted'],
    handwritten_vs_else=results['handwritten_vs_else.h5']['predicted'],
    handwritten_vs_no_code=results['handwritten_vs_no_code.h5']['predicted'],
    all_four=results['all_four.h5']['predicted'])
'''
actual = np.load('actual.npz')
results['code_vs_no_code_strict.h5']['actual'] = actual['code_vs_no_code_strict']
results['code_vs_no_code_partially.h5']['actual'] = actual['code_vs_no_code_partially']
results['code_vs_no_code_partially_handwritten.h5']['actual'] = actual['code_vs_no_code_partially_handwritten']
results['handwritten_vs_else.h5']['actual'] = actual['handwritten_vs_else']
results['handwritten_vs_no_code.h5']['actual'] = actual['handwritten_vs_no_code']
results['all_four.h5']['actual'] = actual['all_four']

predicted = np.load('predicted.npz')
results['code_vs_no_code_strict.h5']['predicted'] = predicted['code_vs_no_code_strict']
results['code_vs_no_code_partially.h5']['predicted'] = predicted['code_vs_no_code_partially']
results['code_vs_no_code_partially_handwritten.h5']['predicted'] = predicted['code_vs_no_code_partially_handwritten']
results['handwritten_vs_else.h5']['predicted'] = predicted['handwritten_vs_else']
results['handwritten_vs_no_code.h5']['predicted'] = predicted['handwritten_vs_no_code']
results['all_four.h5']['predicted'] = predicted['all_four']

y=np.zeros(1000)
y[985:] = 1
p=np.zeros(1000)
p[-1] = 1
p[0] = 1
ps = precision_score(y, p)
rs = recall_score(y, p)
print ps,rs
for key in results.keys():
    y_test = results[key]['actual']
    y_pred = results[key]['predicted']
    cnf_matrix = confusion_matrix(y_test, y_pred)

    average = 'binary'
    if 'all_four' in key:
        average = 'weighted'
    ps = precision_score(y_test, y_pred,average=average)
    rs = recall_score(y_test, y_pred,average=average)

    cnf_matrix /= 5
    plt.figure()

    print key, '{0:.3f}'.format(ps),'&','{0:.3f}'.format(rs)
    if key == 'all_four.h5':
        plot_confusion_matrix(cnf_matrix, classes=four,
                      title='Confusion Matrix: 5 Fold Cross Validation')

    else:
        plot_confusion_matrix(cnf_matrix, classes=two,
                      title='Confusion Matrix: 5 Fold Cross Validation')

    plt.savefig(key.replace('h5','png'))

