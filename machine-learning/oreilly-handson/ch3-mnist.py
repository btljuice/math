from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

random_seed = 42
mnist = fetch_mldata('MNIST original')
X = mnist["data"]
y = mnist["target"]


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row:(row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=matplotlib.cm.binary, **options)
    plt.axis("off")


def plot_digit(x):
    digit_image = x.reshape(28, 28)
    plt.imshow(
        digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")


def kfold_classifier_cross_val_score(clf, X, y):
    skfolds = StratifiedKFold(n_splits=3, random_state=random_seed)
    results = []
    for train_index, test_index in skfolds.split(X, y):
        clone_clf = clone(clf)
        clone_clf.fit(X[train_index], y[train_index])
        y_pred = clone_clf.predict(X[test_index])
        nb_correct = sum(y_pred == y[test_index])
        results.append(nb_correct / len(y_pred))
    return results


def plot_precision_recall_vs_threshold(y, scores):
    p, r, t = precision_recall_curve(y, scores)
    plt.plot(t, p[:-1], "b--", label="Precision")
    plt.plot(t, r[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


def plot_precision_vs_recall(y, scores, options={}):
    p, r, t = precision_recall_curve(y, scores)
    plt.plot(r, p, **options)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0, 1])
    plt.ylim([0, 1])


def plot_roc_curve(y, scores, options):
    fpr, tpr, t = roc_curve(y, scores)
    plt.plot(fpr, tpr, **options)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate = FP/(FP + TN)')
    plt.ylabel('recall=True Positive Rate= TP/(TP+FN)')


def print_splits(split, data):
    for train_index, test_index in split.split(data):
        print("Train : ", sum(train_index), ", Test: ", sum(test_index))

# Author's note: the mnist dataset is already configured to be split as:
# - first 60000 images is the training set
# - last  10000 images is the test set
X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]
some_digit_index = 36000
some_digit = X[some_digit_index]

# Let's control randomness
np.random.seed(random_seed)

# Shuffle training data
# Author's note: Some algorithm are sensible the order of the
# data when they are trained.
shuffle_index = np.random.permutation(60000)
X_train = X_train[shuffle_index]
y_train = y_train[shuffle_index]


def binary_classification_section():
    # 1. Training a binary classifier
    # Train only to identify a 5 vs NOT a 5
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    # Try a Stochastic Gradient Descent Classifier
    # set random_state to a fixed value to settle the randomness during
    # tests
    sgd_clf = SGDClassifier(random_state=random_seed)
    sgd_clf.fit(X_train, y_train_5)

    # Cross-Validation
    print('kfold_classifier_cross_val_score = ',
          kfold_classifier_cross_val_score(
              sgd_clf, X_train,
              y_train_5))  # prints 0.9502, 0.96565 and 0.96495
    print('sdg_clf=',
          cross_val_score(
              sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
    print('never5 =',
          cross_val_score(
              Never5Classifier(), X_train, y_train_5, cv=3,
              scoring="accuracy"))

    # Calculate the confusion matrix
    #  ------------|Predicted Class|
    # |Actual class| Non-5  |  5   |
    # |      Non-5 |  TN    |  FP  |
    # |          5 |  FN    |  TP  |
    # False Positive = Type I error
    # False Negative = Type II error
    # precision = TP / (TP + FP)
    #    Accuracy of the positive predictions
    #    Measures how "useful the positive/detected results are"
    # recall = TP / (TP + FN)
    #    Measure  how "complete the positive/deteted results are"
    # f1_score = 2 / (precision^-1 + recall^-1)
    #    Gives a global score value for the two measures
    #
    # GUESS: these terms seems coined by the medical field
    #
    # Examples:
    # - Detect video safe for kids.
    #   * Want a 'high precision'. No xxx movies for kids.
    #   * Don't care much about 'recall'. A couple of kid movies could be missed
    # - Detect shoplifters on surveillances image.
    #   * Want a 'high recall' (99%)
    #   * Don't care much about a low precision (30%)
    #
    # "precision/recall tradeoff": usually increasing one will reduce the other.
    y_train_pred = cross_val_predict(
        sgd_clf, X_train, y_train_5, cv=3)  # returns predictions True,False
    cm = confusion_matrix(y_train_5, y_train_pred)  # [ TN, FP ; FN TP ]
    precision = cm[1][1] / (cm[1][1] + cm[0][1])
    recall = cm[1][1] / (cm[1][1] + cm[1][0])
    print("(default) precision=", precision_score(y_train_5, y_train_pred))
    print("(default) recall=", recall_score(y_train_5, y_train_pred))
    print("(default) f1_score=", f1_score(y_train_5, y_train_pred))

    # Plot precision/recall vs threshold or precision vs recall
    y_scores_sgd = cross_val_predict(
        sgd_clf, X_train, y_train_5, cv=3,
        method="decision_function")  # returns scores
    plot_precision_recall_vs_threshold(y_train_5, y_scores_sgd)
    plt.show()
    plot_precision_vs_recall(y_train_5, y_scores_sgd)
    plt.show()

    # Ex. We aim for a 90% precision. Looking at the graph, we identify
    # that the need threshold is about 70000.
    y_train_pred_90precision = (y_scores_sgd > 70000)
    print("(90%p) precision=",
          precision_score(y_train_5, y_train_pred_90precision))
    print("(90%p) recall=", recall_score(y_train_5, y_train_pred_90precision))

    # Aiming at a 99% precision, but at what recall price?
    y_train_pred_99precision = (y_scores_sgd > 500000)
    print("(99%p) precision=",
          precision_score(y_train_5, y_train_pred_99precision))
    print("(99%p) recall=", recall_score(y_train_5, y_train_pred_99precision))

    # ROC : Receiver operation characteristics
    # True Positive Rate = sensitivity = Recall = TP / (TP + FN)
    # False Positive Rate = 1 -specificity = FP / (TN + FP)
    # Author's note: We want to stay as far as possible of the dot line
    # AUC : Area under the curve
    # Author's note:
    # Prefer Precision-Recall curve when:
    # - Positive class is rare
    # - when you care a lot about false positive
    # Use ROC in contrary cases.

    forest_clf = RandomForestClassifier(random_state=random_seed)
    y_probas_forest = cross_val_predict(
        forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1]  # prob. of positive class
    plot_roc_curve(y_train_5, y_scores_sgd, {"label": "SGD"})
    plot_roc_curve(y_train_5, y_scores_forest, {"label": "Random Forest"})
    plt.legend(loc="lower right")
    plt.show()
    print("sgd auc = ", roc_auc_score(y_train_5, y_scores_sgd))
    print("rfo auc = ", roc_auc_score(y_train_5, y_scores_forest))

    print("           (sgd) vs (rfo)")
    print("precision = %.2f vs %.2f" %
          (precision_score(y_train_5, y_train_pred),
           precision_score(y_train_5, (y_scores_forest > .5))))
    print("recall    = %.2f vs %.2f" % (recall_score(y_train_5, y_train_pred),
                                        recall_score(y_train_5,
                                                     (y_scores_forest > .5))))
    print("f1_score  = %.2f vs %.2f" % (f1_score(y_train_5, y_train_pred),
                                        f1_score(y_train_5,
                                                 (y_scores_forest > .5))))

    plot_precision_vs_recall(y_train_5, y_scores_sgd, {'label': 'sgd'})
    plot_precision_vs_recall(y_train_5, y_scores_forest,
                             {'label': 'rand. forest'})
    plt.legend(loc="lower left")
    plt.show()


def multi_classification_section():
    # Author's note: Use Binary Classicators for multiclass
    # For N class
    # * (One vs All strategy)
    #   - Have N Binary classificators.
    #   - Each binary classificator will try to identify one
    #     single digit (e.g. 5) vs the rest.
    #   - Choose class from the binary classificator with the best score.
    # * (One vs One strategy)
    #   - N*(N-1)/2 classifier, distinguish between:
    #     0 vs 1, 0 vs 2, ...
    #     1 vs 2, 1 vs 3, ...
    #     ...
    #   - Choose the one who wins most duels.
    #   - Train each binary classifier with less data: train only
    #     w/ the 2 digits of the duel.
    #     Good for model that scales badly with data size.
    ova_clf = SGDClassifier(random_state=random_seed)  # by default, some bin. clf will use OvA when multiclasses are detected
    ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=random_seed))
    forest_clf = RandomForestClassifier(random_state=random_seed)
    ova_clf.fit(X_train, y_train)
    ovo_clf.fit(X_train, y_train)
    forest_clf.fit(X_train, y_train)
    print("for image ", some_digit_index, ":")
    print("ova prediction= ", ova_clf.predict([some_digit]))  # 5
    print("ova scores=", ova_clf.decision_function([some_digit]))
    print("ova argmax(scores)=", np.argmax(ova_clf.decision_function([some_digit])))
    print("ova classes=", ova_clf.classes_)
    print("ovo prediction= ", ovo_clf.predict([some_digit]))  # 5
    print("ovo scores=", ovo_clf.decision_function([some_digit]))
    print("ovo argmax(scores)=", np.argmax(ovo_clf.decision_function([some_digit])))
    print("forest probs=", forest_clf.predict_proba([some_digit]))
    print("----")

    # Cross validation
    print("ova x-val score=",
          cross_val_score(ova_clf, X_train, y_train, cv=3, scoring="accuracy"))
    # Author's note: Simply applying a standard scaler will give a 5% bonus on accuracy
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    print("ova x-val score (scaled)=",
          cross_val_score(ova_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))
    y_train_pred = cross_val_predict(ova_clf, X_train_scaled, y_train, cv=3)

    # Display the confussius matrix
    conf_mx = confusion_matrix(y_train, y_train_pred)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()

    # Display confusion matrix only for errors
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()

    # Diplay some errors
    cl_a, cl_b = 3, 5
    X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
    X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
    X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
    X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

    plt.figure(figsize=(8, 8))
    plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
    plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
    plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
    plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
    plt.show()


def multilabel_classification_section():
    y_train_large = (y_train >= 7)
    y_train_odd = (y_train % 2 == 1)
    y_multilabel = np.c_[y_train_large, y_train_odd]  # like np.column_stack
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel)
    print("knn multilabel prediction is 5 : large,odd=", knn_clf.predict([some_digit]))
    # Do not execute as it might take hours to complete.
    # Compute score of model
    # y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=1)
    # f1_score(y_train, y_train_knn_pred, average="macro")

def multioutput_classification_section():
    noise = np.random.randint(0, 100, (len(X_train), 784))
    X_train_mod = X_train + noise
    noise = np.random.randint(0, 100, (len(X_test), 784))
    X_test_mod = X_test + noise
    y_train_mod = X_train  # w/o noise
    y_test_mod = y_train   # w/o noise
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train_mod, y_train_mod)
    # Show the two digits
    dirty_digit = X_train_mod[some_digit_index]
    clean_digit = knn_clf.predict([dirty_digit])
    plt.subplot(121)
    plot_digit(dirty_digit)
    plt.subplot(122)
    plot_digit(clean_digit)
    plt.show()


