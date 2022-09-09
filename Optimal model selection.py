import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier as XGBC
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc

# Data preparation
"""
The dataset generated during the current study is not publicly available due to restrictions in the ethical permit, 
but may be available from the corresponding author on reasonable request.
"""
data = "Available from the corresponding author on reasonable request"


# Parameter optimization and model comparison on on model development dataset
# RF
param = {"n_estimators":range(100,5001),
         "criterion":('gini','entropy'),
         "max_depth":range(6,13),
         "max_features":('sqrt','log2'),
         "min_samples_split":range(2,11),
         "min_samples_leaf":range(1,6),
         "warm_start":('False','True'),
         "ccp_alpha":[i/100.0 for i in range(0,11)],
         "min_impurity_decrease":[i/100.0 for i in range(1,6)]
        }

# SVM
param2 = [{"kernel": ['poly'],"C":(0.001,0.01,0.1,1,10,100,100,1000),'degree': [2, 3,4,5]},
          {"kernel": ['sigmoid'],"C":[0.001,0.01,0.1,1,10,100,100,1000]},
         {'kernel': ['rbf'], 'C': [0.001,0.01,0.1,1, 10, 100, 100,1000], 'gamma':['scale','auto']}
         ]

# LR
param3 = [{"solver": ['lbfgs'], "penalty": ['l2', 'none'],
           "C": [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]},
          {"solver": ['sag'], "penalty": ['l2', 'none'],
           "C": [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]},
          {"solver": ['saga'], "penalty": ['l1', 'l2', 'none'],
           "C": [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]},
          {"solver": ['saga'], "penalty": ['elasticnet'],
           "l1_ratio": [i / 10.0 for i in range(1, 10)],
           "C": [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]},
          {"solver": ['liblinear'], "penalty": ['l1', 'l2'],
           "C": [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]},
          {"solver": ['newton-cg'], "penalty": ['l2', 'none'],
           "C": [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]},
          ]

# ANN
param4 = {"solver":['adam','sgd'],
          "hidden_layer_sizes": [(32),(64),(128),(256),(512),(1024),
                                 (128,64),(256,128),(512,256),
                                 (128,128,64),(256,256,128),(512,512,256)
                                ],
         "activation":['logistic', 'tanh', 'relu'],
         "alpha":[0.1,0.01,0.001,0.0001,0.00001]
          }
# XGBoost
param5 = {"learning_rate":[0.1,0.01,0.001,0.0001,0.00001],
          "max_depth":range(6,13),
          "n_estimators":range(100,5001),
          "reg_alpha":[i/10.0 for i in range(0,10)],
          "reg_lambda":[i/10.0 for i in range(1,10)],
          "subsample":[i/10.0 for i in range(5,11)],
          "colsample_bytree":[i/10.0 for i in range(5,9)],
          "colsample_bylevel":[i/10.0 for i in range(5,9)]
          }

y_pred_train_rf = []
y_pred_train_svm = []
y_pred_train_lr = []
y_pred_train_ann = []
y_pred_train_xgboost = []

y_pred_rf = []
y_pred_svm = []
y_pred_lr = []
y_pred_ann = []
y_pred_xgboost = []

y_probas_rf_train = []
y_probas_svm_train = []
y_probas_lr_train = []
y_probas_ann_train = []
y_probas_xgboost_train = []

y_probas_rf = []
y_probas_svm = []
y_probas_lr = []
y_probas_ann = []
y_probas_xgboost = []

y_train_list = []
y_test_list = []



for i in range(100):
    x_train, x_test, y_train, y_test = train_test_split(data, data, random_state=None, test_size=0.2)

    y_train_list.append(y_train)
    y_test_list.append(y_test)

    print("Current repeat times:", i + 1)

    rfcmodel = RFC(n_estimators=150, criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=1,
                   min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True,
                   oob_score=True, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None,
                   ccp_alpha=0.0, max_samples=None)
    gs1 = RandomizedSearchCV(estimator=rfcmodel, param_distributions=param, cv=10, n_iter=50, n_jobs=-1)
    gs1.fit(x_train, y_train)
    print("gs1.best_score_", gs1.best_score_)
    print("gs1.best_params_", gs1.best_params_)
    sc_train1 = gs1.score(x_train, y_train)
    sc1 = gs1.score(x_test, y_test)
    y_pred_train = gs1.predict(x_train)
    y_pred = gs1.predict(x_test)
    y_pred_train_rf.append(y_pred_train)
    y_pred_rf.append(y_pred)
    probas_train = gs1.predict_proba(x_train)
    y_probas_rf_train.append(probas_train)
    probas = gs1.predict_proba(x_test)
    y_probas_rf.append(probas)
    print("RF_train: acc", sc_train1)
    print("RF_test : acc", sc1)
    print("")

    SVMC = SVC(kernel="poly", C=15, probability=True)
    gs2 = RandomizedSearchCV(estimator=SVMC, param_distributions=param2, cv=10, n_iter=50, n_jobs=-1)
    gs2.fit(x_train, y_train)
    print("gs2.best_score_", gs2.best_score_)
    print("gs2.best_params_", gs2.best_params_)
    sc_train1 = gs2.score(x_train, y_train)
    sc1 = gs2.score(x_test, y_test)
    y_pred_train = gs2.predict(x_train)
    y_pred = gs2.predict(x_test)
    y_pred_train_svm.append(y_pred_train)
    y_pred_svm.append(y_pred)
    probas_train = gs2.predict_proba(x_train)
    y_probas_svm_train.append(probas_train)
    probas = gs2.predict_proba(x_test)
    y_probas_svm.append(probas)
    print("SVM_train: acc", sc_train1)
    print("SVM_test : acc", sc1)
    print("")

    LRmodel = LR(random_state=None, solver='lbfgs', max_iter=10000, penalty='l2', tol=0.0001)
    gs3 = RandomizedSearchCV(estimator=LRmodel, param_distributions=param3, cv=10, n_iter=50, n_jobs=-1)
    gs3.fit(x_train, y_train)
    print("gs3.best_score_", gs3.best_score_)
    print("gs3.best_params_", gs3.best_params_)
    sc_train1 = gs3.score(x_train, y_train)
    sc1 = gs3.score(x_test, y_test)
    y_pred_train = gs3.predict(x_train)
    y_pred = gs3.predict(x_test)
    y_pred_train_lr.append(y_pred_train)
    y_pred_lr.append(y_pred)
    probas_train = gs3.predict_proba(x_train)
    y_probas_lr_train.append(probas_train)
    probas = gs3.predict_proba(x_test)
    y_probas_lr.append(probas)
    print("LR_train: acc", sc_train1)
    print("LR_test : acc", sc1)
    print("")

    MLPmodel = MLPClassifier(solver='adam', activation='logistic', alpha=0.001, early_stopping=False,
                             hidden_layer_sizes=(128, 128, 64), max_iter=10000, random_state=None, shuffle=True)
    gs4 = RandomizedSearchCV(estimator=MLPmodel, param_distributions=param4, cv=10, n_iter=50, n_jobs=-1)
    gs4.fit(x_train, y_train)
    print("gs4.best_score_", gs4.best_score_)
    print("gs4.best_params_", gs4.best_params_)
    sc_train1 = gs4.score(x_train, y_train)
    sc1 = gs4.score(x_test, y_test)
    y_pred_train = gs4.predict(x_train)
    y_pred = gs4.predict(x_test)
    y_pred_train_ann.append(y_pred_train)
    y_pred_ann.append(y_pred)
    probas_train = gs4.predict_proba(x_train)
    y_probas_ann_train.append(probas_train)
    probas = gs4.predict_proba(x_test)
    y_probas_ann.append(probas)
    print("ANN_train: acc", sc_train1)
    print("ANN_test : acc", sc1)
    print("")

    xgb = XGBC(learning_rate=0.2, n_estimators=800, max_depth=8, min_child_weight=0.5, gamma=0.5, subsample=0.8,
               colsample_bytree=0.8, objective='binary:logistic', scale_pos_weight=1, seed=None, nthread=-1)
    gs5 = RandomizedSearchCV(estimator=xgb, param_distributions=param5, cv=10, n_iter=50, n_jobs=-1)
    gs5.fit(x_train, y_train)
    print("gs5.best_score_", gs5.best_score_)
    print("gs5.best_params_", gs5.best_params_)
    sc_train1 = gs5.score(x_train, y_train)
    sc1 = gs5.score(x_test, y_test)
    y_pred_train = gs5.predict(x_train)
    y_pred = gs5.predict(x_test)
    y_pred_train_xgboost.append(y_pred_train)
    y_pred_xgboost.append(y_pred)
    probas_train = gs5.predict_proba(x_train)
    y_probas_xgboost_train.append(probas_train)
    probas = gs5.predict_proba(x_test)
    y_probas_xgboost.append(probas)
    print("XGBoost_train: acc", sc_train1)
    print("XGBoost_test : acc", sc1)
    print("")

#Model evaluation
def get_index(Y_test, y_probas):
    specificity = []
    au = []
    acc = []
    sensitivity = []
    f1_s = []

    for i in range(100):
        fpr, tpr, thresholds = roc_curve(Y_test.iloc[:, i], y_probas.iloc[:, i], pos_label=1)
        roc_auc = auc(fpr, tpr)

        Y_pred = []
        value = y_probas.iloc[:, i]
        for t in range(len(value)):
            if (value[t] >= 0.5):
                Y_pred.append(1)
            else:
                Y_pred.append(0)

        con_mat = confusion_matrix(Y_test.iloc[:, i], Y_pred)
        tn = con_mat[0][0]
        fp = con_mat[0][1]
        specificity1 = tn / (tn + fp)
        acc1 = accuracy_score(Y_test.iloc[:, i], Y_pred)
        sensitivity1 = recall_score(Y_test.iloc[:, i], Y_pred)
        f1_1 = f1_score(Y_test.iloc[:, i], Y_pred)
        specificity.append(specificity1)
        au.append(roc_auc)
        sensitivity.append(sensitivity1)
        acc.append(acc1)
        f1_s.append(f1_1)
    return au, sensitivity, specificity, f1_s, acc


au, sensitivity, specificity, f1_s , acc = get_index(y_test_list,y_probas_rf)
class_rf = []
class_rf_std = []
class_rf.append(np.mean(au))
class_rf.append(np.mean(sensitivity))
class_rf.append(np.mean(specificity))
class_rf.append(np.mean(f1_s))
class_rf.append(np.mean(acc))
class_rf_std.append(np.std(au))
class_rf_std.append(np.std(sensitivity))
class_rf_std.append(np.std(specificity))
class_rf_std.append(np.std(f1_s))
class_rf_std.append(np.std(acc))
print(class_rf)
print("")

au, sensitivity, specificity, f1_s , acc = get_index(y_test_list,y_probas_svm )
class_svm = []
class_svm_std = []
class_svm.append(np.mean(au))
class_svm.append(np.mean(sensitivity))
class_svm.append(np.mean(specificity))
class_svm.append(np.mean(f1_s))
class_svm.append(np.mean(acc))
class_svm_std.append(np.std(au))
class_svm_std.append(np.std(sensitivity))
class_svm_std.append(np.std(specificity))
class_svm_std.append(np.std(f1_s))
class_svm_std.append(np.std(acc))
print(class_svm)
print("")

au, sensitivity, specificity, f1_s , acc = get_index(y_test_list,y_probas_lr )

class_lr = []
class_lr_std = []
class_lr.append(np.mean(au))
class_lr.append(np.mean(sensitivity))
class_lr.append(np.mean(specificity))
class_lr.append(np.mean(f1_s))
class_lr.append(np.mean(acc))
class_lr_std.append(np.std(au))
class_lr_std.append(np.std(sensitivity))
class_lr_std.append(np.std(specificity))
class_lr_std.append(np.std(f1_s))
class_lr_std.append(np.std(acc))
print(class_lr)
print("")


au, sensitivity, specificity, f1_s , acc = get_index(y_test_list,y_probas_ann)

class_ann = []
class_ann_std = []
class_ann.append(np.mean(au))
class_ann.append(np.mean(sensitivity))
class_ann.append(np.mean(specificity))
class_ann.append(np.mean(f1_s))
class_ann.append(np.mean(acc))
class_ann_std.append(np.std(au))
class_ann_std.append(np.std(sensitivity))
class_ann_std.append(np.std(specificity))
class_ann_std.append(np.std(f1_s))
class_ann_std.append(np.std(acc))
print(class_ann)
print("")

au, sensitivity, specificity, f1_s , acc = get_index(y_test_list,y_probas_xgboost )
class_xgboost = []
class_xgboost_std = []
class_xgboost.append(np.mean(au))
class_xgboost.append(np.mean(sensitivity))
class_xgboost.append(np.mean(specificity))
class_xgboost.append(np.mean(f1_s))
class_xgboost.append(np.mean(acc))
class_xgboost_std.append(np.std(au))
class_xgboost_std.append(np.std(sensitivity))
class_xgboost_std.append(np.std(specificity))
class_xgboost_std.append(np.std(f1_s))
class_xgboost_std.append(np.std(acc))
print(class_xgboost)
print("")



excel_rf = []
excel_svm = []
excel_lr = []
excel_ann = []
excel_xgb = []
for i in range(5):
    if(i<1):
        str1 = str(format(class_rf[i],'.3f')) +"±" + str(format(class_rf_std[i],'.3f'))
        str2 = str(format(class_svm[i],'.3f')) +"±" + str(format(class_svm_std[i],'.3f'))
        str3 = str(format(class_lr[i],'.3f')) +"±" + str(format(class_lr_std[i],'.3f'))
        str4 = str(format(class_ann[i],'.3f')) +"±" + str(format(class_ann_std[i],'.3f'))
        str5 = str(format(class_xgboost[i],'.3f')) +"±" + str(format(class_xgboost_std[i],'.3f'))
    else:
        str1 = str(format(class_rf[i]*100,'.2f')) +"±" + str(format(class_rf_std[i]*100,'.2f'))
        str2 = str(format(class_svm[i]*100,'.2f')) +"±" + str(format(class_svm_std[i]*100,'.2f'))
        str3 = str(format(class_lr[i]*100,'.2f')) +"±" + str(format(class_lr_std[i]*100,'.2f'))
        str4 = str(format(class_ann[i]*100,'.2f')) +"±" + str(format(class_ann_std[i]*100,'.2f'))
        str5 = str(format(class_xgboost[i]*100,'.2f')) +"±" + str(format(class_xgboost_std[i]*100,'.2f'))
    excel_rf.append(str1)
    excel_svm.append(str2)
    excel_lr.append(str3)
    excel_ann.append(str4)
    excel_xgb.append(str5)

evaluation_data = []
evaluation_data.append(excel_rf)
evaluation_data.append(excel_svm)
evaluation_data.append(excel_lr)
evaluation_data.append(excel_ann)
evaluation_data.append(excel_xgb)


evaluation_data_show = pd.DataFrame(evaluation_data, columns=['AUC','SEN (%)','SPE (%)','F1 (%)','ACC (%)'], index = ['RF','SVM','LR','ANN','XGB'])
print(evaluation_data_show)
