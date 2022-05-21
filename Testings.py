import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import learning_curve, ShuffleSplit
import pandas as pd
import seaborn as sb

def Model_Testing(model, Xtest,ytest,Classes, xtrain , ytrain, combined , all_classes):


    predicted = model.predict(Xtest)
    #fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    #shfle = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

    print( "Accuracy Score : " + str(round(accuracy_score(ytest,predicted)*100 , 2)))
    print("Recall Score : " + str(recall_score(ytest, predicted , average='macro')))
    print("Precision : " + str(precision_score(ytest, predicted , average='macro')))
    print("f1 Score : " + str(f1_score(ytest, predicted , average='macro')))


    # displaying heatmap

    #plot_learning_curve(model,model2, model3,  "Learning Curve", combined , all_classes, ylim=(0.6, 1.01),
     #                   cv=shfle, n_jobs=4).show()
    #Confusion =   confusion_matrix(ytest , predicted , labels=Classes )
    #plot_confusion_matrix(Confusion,Classes)
    #plt.figure(figsize=(10, 15))
    #cmtx = np.round(pd.DataFrame(
     #   confusion_matrix(ytest , predicted , labels=Classes,normalize="true" ),
    #) , 2)
    conf_mat = confusion_matrix(ytest, predicted,labels=Classes)

    conf_mat_normalized = np.round(pd.DataFrame((conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis])) , 2)

    sb.heatmap(conf_mat_normalized, annot=True, xticklabels=Classes, yticklabels=Classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
   #print(cmtx)
    #cmtx.to_csv("./ASL_RF_CM.csv")


    #np.interp.plot_confusion_matrix()

def plot_learning_curve(estimator, estimator2, estimator3, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_ylim([0.7,1.01])
    axes[1].set_ylim([0.7, 1.01])
    axes[2].set_ylim([0.7, 1.01])

    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Score")

    axes[2].set_xlabel("Training examples")
    axes[2].set_ylabel("Score")

    train_sizes1, train_scores1, test_scores1, fit_times1, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean1 = np.mean(train_scores1, axis=1)
    train_scores_std1 = np.std(train_scores1, axis=1)
    test_scores_mean1 = np.mean(test_scores1, axis=1)
    test_scores_std1= np.std(test_scores1, axis=1)
    fit_times_mean1 = np.mean(fit_times1, axis=1)
    fit_times_std1 = np.std(fit_times1, axis=1)

    train_sizes2, train_scores2, test_scores2, fit_times2, _ = \
        learning_curve(estimator2, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean2 = np.mean(train_scores2, axis=1)
    train_scores_std2 = np.std(train_scores2, axis=1)
    test_scores_mean2 = np.mean(test_scores2, axis=1)
    test_scores_std2 = np.std(test_scores2, axis=1)
    fit_times_mean2 = np.mean(fit_times2, axis=1)
    fit_times_std2 = np.std(fit_times2, axis=1)

    train_sizes3, train_scores3, test_scores3, fit_times3, _ = \
        learning_curve(estimator3, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean3 = np.mean(train_scores3, axis=1)
    train_scores_std3 = np.std(train_scores3, axis=1)
    test_scores_mean3 = np.mean(test_scores3, axis=1)
    test_scores_std3 = np.std(test_scores3, axis=1)
    fit_times_mean3 = np.mean(fit_times3, axis=1)
    fit_times_std3 = np.std(fit_times3, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean1 - train_scores_std1,
                         train_scores_mean1 + train_scores_std1, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean1 - test_scores_std1,
                         test_scores_mean1 + test_scores_std1, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean1, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean1, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    plt.ylim([0.7, 1.01])

    # Plot learning curve
    axes[1].grid()
    axes[1].fill_between(train_sizes, train_scores_mean2 - train_scores_std2,
                         train_scores_mean2 + train_scores_std2, alpha=0.1,
                         color="r")
    axes[1].fill_between(train_sizes, test_scores_mean2 - test_scores_std2,
                         test_scores_mean2 + test_scores_std2, alpha=0.1,
                         color="g")
    axes[1].plot(train_sizes, train_scores_mean2, 'o-', color="r",
                 label="Training score")
    axes[1].plot(train_sizes, test_scores_mean2, 'o-', color="g",
                 label="Cross-validation score")
    axes[1].legend(loc="best")

    plt.ylim([0.7, 1.01])


    # Plot learning curve
    axes[2].grid()
    axes[2].fill_between(train_sizes, train_scores_mean3 - train_scores_std3,
                         train_scores_mean3 + train_scores_std3, alpha=0.1,
                         color="r")
    axes[2].fill_between(train_sizes, test_scores_mean3 - test_scores_std3,
                         test_scores_mean3 + test_scores_std3, alpha=0.1,
                         color="g")
    axes[2].plot(train_sizes, train_scores_mean3, 'o-', color="r",
                 label="Training score")
    axes[2].plot(train_sizes, test_scores_mean3, 'o-', color="g",
                 label="Cross-validation score")
    axes[2].legend(loc="best")

    plt.ylim([0.7,1.01])

    # Plot n_samples vs fit_times
    #axes[1].grid()
    #axes[1].plot(train_sizes, fit_times_mean, 'o-')
    #axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
     #                    fit_times_mean + fit_times_std, alpha=0.1)
    #axes[1].set_xlabel("Training examples")
    #axes[1].set_ylabel("fit_times")
    #axes[1].set_title("Scalability of the model")


    # Plot fit_time vs score
    #axes[2].grid()
    #axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    #axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
     #                    test_scores_mean + test_scores_std, alpha=0.1)
    #axes[2].set_xlabel("fit_times")
    #axes[2].set_ylabel("Score")
    #axes[2].set_title("Performance of the model")

    return plt

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(np.round(cm , 2))

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

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()