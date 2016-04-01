__author__ = 'Prudhvi Badri'

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import metrics
from ggplot import *
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

X = pd.read_csv('Electronics_Output/Text_Features.csv')
X = X[0:24000]
X.fillna(0,inplace=True)
X.drop(['REVIEW_TEXT'],axis=1,inplace=True)
# print X.head()
Y = X.pop('CLASS')  #store class label in Y
numeric_variables = list(X.dtypes[X.dtypes!="object"].index)
model = RandomForestClassifier(n_estimators= 100 ,criterion="gini",  n_jobs= 2)
model2 = GradientBoostingClassifier()
model.fit(X[numeric_variables],Y)
importances = model.feature_importances_
# print importances

predicted_1 = cross_validation.cross_val_predict(model, X[numeric_variables], Y, cv=10)
predicted_2 = cross_validation.cross_val_predict(model2, X[numeric_variables], Y, cv=10)

print "Classification of Helpfulness reviews in Electronics reviews dataset"
print "--------------------------------------------------------------"
print "Random Forest Accuracy: ", metrics.accuracy_score(Y, predicted_1)
print "Confusion Matrix For Random Forest Classifier"
print metrics.confusion_matrix(Y,predicted_1)
print "AUC Score : ",metrics.roc_auc_score(Y,predicted_1)
print "Recall : ",metrics.recall_score(Y,predicted_1)
print "Average Precision Score : ",metrics.average_precision_score(Y,predicted_1)
print " "
print "############################################"
print " "
print "Gradient Boosting Accuracy: ", metrics.accuracy_score(Y, predicted_2)
print "Confusion Matrix For Gradient Boosting Classifier"
print metrics.confusion_matrix(Y,predicted_2)
print "AUC Score : ",metrics.roc_auc_score(Y,predicted_2)
print "Recall : ",metrics.recall_score(Y,predicted_2)
print "Average Precision Score : ",metrics.average_precision_score(Y,predicted_2)


fpr, tpr, _ = metrics.roc_curve(Y, predicted_2)
print predicted_1
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
print df
print ggplot(df, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed')
auc = metrics.auc(fpr,tpr)
print ggplot(df, aes(x='fpr', ymin=0, ymax='tpr')) + geom_area(alpha=0.2) + geom_line(aes(y='tpr')) + ggtitle("ROC Curve w/ AUC=%s" % str(auc))
