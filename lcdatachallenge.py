from petl import fromcsv,look,cutout,select,fieldmap,tocsv,merge,split,sort
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn import svm
from collections import OrderedDict
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
import csv
import string
import re
"""
ETL function used for feature engineering.
petl library has been used
from petl import fromcsv,look,cutout,select,fieldmap,tocsv,merge,split,sort
returns two csv files,1.for features 2.for Labels
"""
def dataPreProcessing(fileName):
	inputData = fromcsv(fileName)
	table1 = cutout(inputData,'member_id','grade','sub_grade','emp_title','url','desc','title','accept_d','exp_d','list_d','issue_d','purpose','addr_city','addr_state','earliest_cr_line','last_pymnt_d','next_pymnt_d','last_credit_pull_d')
	table2 = select(table1,lambda i: i['term'] == ' 36 months' and i['loan_status'] is not "")
	labelMapping = OrderedDict()
	labelMapping['loan_status'] = 'loan_status'
	labelMapping['id'] = 'id'
	table6 = fieldmap(table2,labelMapping)
	table8 = sort(table6,'id')
	table10 = cutout(table8,'id')
	mappings = OrderedDict()
	mappings['id'] = 'id'
	mappings['home_ownership'] = 'ownership',{'MORTGAGE':'-1','RENT':'0','OWN':'1'}
	mappings['emp_length'] = 'empLength',{'n/a':0}
	mappings['is_inc_v'] = 'verificationStatus',{'Source Verified':1,'Verified':0,'Not Verified':-1}
	mappings['pymnt_plan'] = 'paymentPlan',{'n':0,'y':1}
	mappings['initial_list_status'] = 'listStatus',{'f':0,'w':1}
	table3 = fieldmap(table2,mappings)
	table4 = cutout(table2,'home_ownership','is_inc_v','pymnt_plan','initial_list_status','term','loan_status')
	table5 = merge(table3,table4,key = 'id')
	table7 = sort(table5,'id')
	table9 = cutout(table7,'id')
	featureFileCsv = tocsv(table9,'featureFileCsv.csv')
	labelsFileCsv = tocsv(table10,'labelsFileCsv.csv')
	return featureFileCsv,labelsFileCsv

"""
Function to remove non numeric characters from features
"""
def removeCharacters(s):
	n = 0
	for letters in s:
		if not letters.isdigit():
			return n
		else:
			n = n * 10 + int(letters)
	return n 

"""
Function to convert featureFileCsv.csv into a valid CSR matrix to be fed into SVM Classifier
returns featureMatrix of type float
"""	
def getDataList(fileName):	
	customerRecords = []
	with open(fileName) as a:
		fields = map(str.rstrip, a)		
		for field in fields[1:]:
	   		customerRecords.append(field)
	   	featureList = list(csv.reader(customerRecords))	
	for elements in featureList:
		i = 0
		for fields in elements:
			fields = fields.split(' ')[0]
	featureMatrix = []
	for elements in featureList:
		tempList = [removeCharacters(s) for s in elements]
		featureMatrix.append(tempList)
	
	return csr_matrix(featureMatrix,dtype=float)


"""
Function to create labels out of data provided
Label 1 for loan_status = Fully Paid
Label 0 for loan_status = any other value
"""
def getTrueLabels(fileName):
	customerLabels = []
	with open(fileName) as a:
		fields = map(str.rstrip, a)
		for field in fields[1:]:
			customerLabels.append(field)
	trueLabels = np.zeros((len(customerLabels)),dtype = np.int)
	for status in range(len(customerLabels)):
		if (customerLabels[status] == 'Fully Paid'):
			trueLabels[status] = 1
		else:
			trueLabels[status] = 0
	return trueLabels
"""
Function to call SVM classifier,train the model and check accuracy through cross validation
returns average accuracy.
"""

def do_cross_validation(X, y, n_folds):
	kf = KFold(len(y), n_folds=n_folds)
	accuracy = []
	for train_idx, test_idx in kf:
		clf = svm.LinearSVC()
		clf.fit(X[train_idx], y[train_idx])
		predicted = clf.predict(X[test_idx])
		accuracy.append(accuracy_score(y[test_idx], predicted))
	avg = np.mean(accuracy)
	return avg*100
"""
Main Function
"""
def main():

	fileName = 'loan3b.csv'
	dataPreProcessing(fileName)
	featureMatrix = getDataList('featureFileCsv.csv')
	trueLabels = getTrueLabels('labelsFileCsv.csv')
	average_acc = do_cross_validation(featureMatrix,trueLabels,n_folds=7)
	print average_acc

""" Invoking Main Function """		
if __name__=="__main__":
	main()
