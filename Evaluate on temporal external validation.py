import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import roc_curve, auc

# Data preparation
"""
The dataset generated during the current study is not publicly available due to restrictions in the ethical permit, 
but may be available from the corresponding author on reasonable request.
"""
data = "Available from the corresponding author on reasonable request"


# Parameter optimization on model development dataset with RF model
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

rfcmodel = RFC(n_estimators=150, criterion='gini', max_depth=10, min_samples_split=2, min_samples_leaf=1,
               min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
               bootstrap=True, oob_score=True,n_jobs=-1, random_state=None,
               verbose=0, warm_start=False,class_weight=None, ccp_alpha=0.0, max_samples=None)
rf_model = RandomizedSearchCV(estimator=rfcmodel, param_distributions=param, cv=10, n_iter=50, n_jobs=-1)
rf_model.fit(data, data)
print("gs1.best_params_", rf_model.best_params_)

# Test on temporal external validation dataset
rf_probs= rf_model.predict_proba(data)
fpr, tpr, thresholds = roc_curve(data, rf_probs[:,1] ,pos_label=1)
roc_auc = auc(fpr, tpr)
print("AUC of temporary external validation dataset:",roc_auc)