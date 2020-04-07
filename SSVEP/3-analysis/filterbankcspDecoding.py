###############################################
####Filter Bank -> CSP -> Classification########
###############################################

"""
1. **Feature Extraction**: Filter bank band pass channels will be used as main features.  To extract the features, we are going to use a **filter-bank of band-pass channels**, namely 4-8Hz, 6-10Hz, 8-12Hz, etc.
2. **Feature Selection**: Common Spatial Pattern will help find the components that exhibit maximum variances which helps project the data into most discriminating features (i.e., features with maximum variance).  Each pair of band-pass and CSP filters computes the CSP features, which are specific to the band-pass frequency range.
3. **Classification**
"""

import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

from mne import Epochs, find_events
from mne.decoding import CSP, Vectorizer, Scaler, SPoC

from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances

from collections import OrderedDict
import helper as helper

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sns


"""
- **CSP + Classifier** :  Common Spatial Patterns + Regularized Linear Discriminat Analysis. This is a very common EEG analysis pipeline.
- **Cov + MDM**: Covariance + MDM. A very simple, yet effective (for low channel count), Riemannian geometry classifier.
- **Cov + TS** :  Covariance + Tangent space mapping. One of the most reliable Riemannian geometry-based pipelines.

Evaluation is done through cross-validation, with area-under-the-curve (AUC) as metric (AUC is probably the best metric for binary and unbalanced classification problem)

*Note: because we're doing machine learning here, the following cell may take a while to complete*

*Note: Scikit-learn API provides functionality to chain transformers and estimators by using sklearn.pipeline.Pipeline. We can construct decoding pipelines and perform cross-validation and grid-search. However scikit-learn transformers and estimators generally expect 2D data (n_samples * n_features), whereas MNE transformers typically output data with a higher dimensionality (e.g. n_samples * n_channels * n_times). A Vectorizer or Covariances or CSP therefore needs to be applied between the MNE and the scikit-learn steps.
"""
def decode(epoch):

    epoch.pick_types(eeg=True)
    X = epoch.get_data() * 1e6 #n_epochs * n_channel * n_time_samples  
    cov_X = Covariances().transform(X)
    #print(cov_X)
    #print(X)
     #CSP will take in data in this form and create features of 2d
    y = epoch.events[:, -1]
    y = label_binarize(y,classes=[1,2,3]) #one-hot encoding

    #cv = KFold(n_splits=10, random_state=None)

    cv = StratifiedShuffleSplit(n_splits=30, test_size=0.25)

    # #classification with Minimum distance to mean
    # mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
    # scores = cross_val_score(mdm, cov_X, y, cv=cv, n_jobs=1)

    # class_balance = np.mean(y == y[0])
    # class_balance = max(class_balance, 1. - class_balance)
    # print("MDM Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
    #                                                           class_balance))
    
    
    # #classification with Tangent Space Logistic Regression
    # clf = TSclassifier()
    # # Use scikit-learn Pipeline with cross_val_score function
    # scores = cross_val_score(clf, cov_X, y, cv=cv, n_jobs=1)

    # # Printing the results
    # class_balance = np.mean(y == y[0])
    # class_balance = max(class_balance, 1. - class_balance)
    # print("Tangent space Classification accuracy: %f / Chance level: %f" %
    #       (np.mean(scores), class_balance))


    # #classification with CSP + logistic Regression
    # lr = LogisticRegression(multi_class='multinomial')  #note the multiclass
    # csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
    # clf = Pipeline([('CSP', csp), ('LogisticRegression', lr)])
    # scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)

    # # Printing the results
    # class_balance = np.mean(y == y[0])
    # class_balance = max(class_balance, 1. - class_balance)
    # print("CSP + LogisticRegression Classification accuracy: %f / Chance level: %f" %
    #       (np.mean(scores), class_balance))

    # #classification with CSP + LDA
    # lda = LDA(shrinkage='auto', solver='eigen') #Regularized LDA #inherently multiclass
    # csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
    # clf = Pipeline([('CSP', csp), ('lda', lda)])
    # scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)

    # # Printing the results
    # class_balance = np.mean(y == y[0])
    # class_balance = max(class_balance, 1. - class_balance)
    # print("CSP + LDA Classification accuracy: %f / Chance level: %f" %
    #       (np.mean(scores), class_balance))

    clfs = OrderedDict()
    
    lda = LDA(shrinkage='auto', solver='eigen') #Regularized LDA
    svc = OneVsRestClassifier(SVC())
    lr = LogisticRegression(multi_class='multinomial')
    knn = KNeighborsClassifier(n_neighbors=3) #you would want to optimize
    nb = GaussianNB()
    rf = RandomForestClassifier(n_estimators=50)
    mdm = MDM()
    ts = TSclassifier()    
    # vec = Vectorizer()    
    # scale = Scaler(epoch.info)  #by default, CSP already does this, but if you use Vectorizer, you hve to do it before Vectorizing
    csp = CSP(n_components=4, reg='ledoit_wolf', log=True)

    # #clfs['Vectorizer + LDA'] = Pipeline([('Scaler', scale), ('Vectorizer', vec), ('Model', lda)])
    clfs['CSP + LDA'] = Pipeline([('CSP', csp), ('lda', lda)])
    clfs['CSP + SVC'] = Pipeline([('CSP', csp), ('svc', svc)])
    clfs['CSP + LR'] = Pipeline([('CSP', csp), ('lr', lr)])
    clfs['CSP + KNN'] = Pipeline([('CSP', csp), ('Model', knn)])
    clfs['CSP + NB'] = Pipeline([('CSP', csp), ('nb', nb)])
    clfs['CSP + RF'] = Pipeline([('CSP', csp), ('rf', rf)])
    clfs['Cov + MDM'] = Pipeline([('Cov', Covariances('oas')), ('mdm', mdm)]) #oas is needed for non-PD matrix
    clfs['Cov + TS'] = Pipeline([('Cov', Covariances('oas')), ('ts', ts)]) #oas is needed for non-PD matrix
    # # #not sure why TS is not working....


    # lda2 = LDA(shrinkage='auto', solver='eigen') #Regularized LDA
    # svc2 = SVC()
    # lr2 = LogisticRegression()
    # knn2 = KNeighborsClassifier(n_neighbors=3) #you would want to optimize
    # nb2 = GaussianNB()
    # rf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    # mdm2 = MDM()
    # ts2 = TangentSpace()
    # vec2 = Vectorizer()
    # scale2 = Scaler(epoch.info)  #by default, CSP already does this, but if you use Vectorizer, you hve to do it before Vectorizing
    # csp2 = CSP(n_components=2, norm_trace=False, log=True) #feature extraction, reg is used when data is not PD (positive definite)

    # #clfs['Vectorizer + LDA'] = Pipeline([('Scaler', scale), ('Vectorizer', vec), ('Model', lda)])
    # clfs['2CSP + LDA'] = Pipeline([('CSP', csp), ('Model', lda2)])
    # clfs['2CSP + SVC'] = Pipeline([('CSP', csp), ('Model', svc2)])
    # clfs['2CSP + LR'] = Pipeline([('CSP', csp), ('Model', lr2)])
    # clfs['2CSP + KNN'] = Pipeline([('CSP', csp), ('Model', knn2)])
    # clfs['2CSP + NB'] = Pipeline([('CSP', csp), ('Model', nb2)])
    # clfs['2CSP + RF'] = Pipeline([('CSP', csp), ('Model', rf2)])
    # clfs['2Cov + MDM'] = Pipeline([('Cov', Covariances('oas')), ('Model', mdm2)]) #oas is needed for non-PD matrix
    # #clfs['Cov + TS'] = Pipeline([('Cov', Covariances('oas')), ('Model', ts)]) #oas is needed for non-PD matrix
    # # #not sure why TS is not working....


    auc = []
    methods = []

    # # define cross validation (i put 10 to reduce time for demo)
    # cv = StratifiedShuffleSplit(n_splits=10, test_size=0.25, 
    #                         random_state=42)

    for m in clfs:
        preds = np.empty(len(y))
        for train, test in cv.split(X, y):
            clfs[m].fit(X[train], y[train])
            preds[test] = clfs[m].predict(X[test])
            acc = accuracy_score(y[test], preds[test])
            print("Accuracy score: {}".format(acc))

        print(classification_report(y[test], preds[test]))

    #     print("+", end="") #to know it's working, no newline
    #     res = cross_val_score(clfs[m], X, y, n_jobs=-1)

    #     # Printing the results
    #     class_balance = np.mean(y == y[0])
    #     class_balance = max(class_balance, 1. - class_balance)
    #     print("CSP + LDDDDDA Classification accuracy: %f / Chance level: %f" %
    #           (np.mean(res), class_balance))

    #     auc.extend(res)
    #     #print(auc)
    #     methods.extend([m]*len(res))
    
    # results = pd.DataFrame(data=auc, columns=['AUC'])
    # results['Method'] = methods
    # print(results.tail())

    # plot(results)

def plot(df):
    figure = plt.figure(figsize=[8,4])
    plt.title("AOC")
    sns.barplot(data=df, x='AUC', y='Method')
    plt.xlim(0.4, 1)    


