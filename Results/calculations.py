import numpy as np
import matplotlib.pyplot as plt
import json

data = json.load(open('all_folds_acc.json'))

keys = ['code_vs_no_code_strict.h5',
'code_vs_no_code_partially.h5',
'code_vs_no_code_partially_handwritten.h5',
'handwritten_vs_else.h5',
'all_four.h5']

latex = open('latex.txt','a')
latex.write('& Mean Accuracy & Median Accuracy \\\\\n')
latex.write('\\hline\n')

for key in keys:
    n_folds = 5
    fileTypes = [1,2,3,4,5]

    folds = data[key]

    fig, ax = plt.subplots()

    index = np.arange(n_folds)
    bar_width = 0.5
    opacity = 0.5
    error_config = {'ecolor': '0.3'}

    plt.bar(index,folds, bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label='R')

    plt.xlabel('Fold')
    plt.ylabel('Percent Error')
    plt.title('Percent Error of Each Fold')
    plt.xticks(index, fileTypes)
    plt.legend()

    plt.tight_layout()
    plt.savefig(key.replace('h5','png'))

    folds = np.array(folds)

    line = key.replace('.h5','').replace('_',' ')+ ' & '+ '{0:.3f}'.format(np.mean(folds)) +' & '+ '{0:.3f}'.format(np.median(folds)) +'\\\\\n'
    latex.write(line)
    latex.write('\\hline\n')
