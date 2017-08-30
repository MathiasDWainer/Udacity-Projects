
# coding: utf-8

# Task 1: Select what features you'll use.
# 
# Task 2: Remove outliers
# 
# Task 3: Create new features
# 
# Task 4: Try different classifiers (Decision Tree, SVC, Naive Bayes
# 
# Task 5: Tune chosen classifier to precission and recall greater than .3
# 
# Task 6: Do the data dump for checker.

# # Poi.id Code

# In[1]:

#Poi.id Code

from IPython.display import Image
import matplotlib.pyplot as plt
import sys
import pickle
from sklearn import preprocessing
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# ## Looking at the Data
# 
# ### Basic Data

# In[2]:

#number of enron employees we are looking at
print ('there are: ',len(data_dict.values()),' employees,')

#number of POIs in dataset
poi_count=0
for x, y in data_dict.items():
  if y['poi']==1:
    poi_count+=1
print ("Person of Interest count:", poi_count)


# ### Finding and removing outliers

# In[3]:

#looking at salary and bonus
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
#plotting below
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

#removing outlier (total) and and re-running code

data = featureFormat(data_dict, features)
#plotting below
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


# ## Creating New Features

# In[4]:

#creating features based on poi to total fraction

#fraction calculator
def computeFraction( poi_messages, all_messages ):
    #convert NaN to 0
    fraction = 0.
    if all_messages == 'NaN':
        return fraction
    if poi_messages == 'NaN':
        return fraction
    fraction = float(poi_messages)/float(all_messages)
    return fraction

#feature creation
submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi
#new feaures have been created
    
#way to print our new features    
'''
for names in data_dict:
    print(data_dict[names]["fraction_from_poi"])    
'''

#graphing feature
features_list = ["poi", "fraction_from_poi", "fraction_to_poi"]    
data = featureFormat(data_dict, features_list)

### plot new features
for point in data:
    fraction_from_poi = point[1]
    fraction_to_poi = point[2]
    plt.scatter( fraction_from_poi, fraction_to_poi )
    if point[0] == 1:
        plt.scatter(fraction_from_poi, fraction_to_poi, color="r", marker="*")
plt.xlabel("fraction from")
plt.ylabel("fraction to")
plt.show()


# There doesn't seem to be any outliers worth removing from our new features.
# 
# ## Feature Selection

# In[5]:

#code from Udacity lessons
#selcting features we want for our annalysis. We want the the highest coefficients

#poi (the label) must be first
features_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi",
                 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']

### split data into training and testing datasets
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
test_classifier(clf, data_dict, features_list, folds = 1000)

#code below from github
importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print ('Feature Ranking: ')
for i in range(16):
    print ("{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]]))


# Precision and recall are around .29. Close but no cigar. Need to remove some features, will start with all 0 coefficients and if that doesn't work i will play around.

# In[6]:

features_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi"]

### split data into training and testing datasets
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
test_classifier(clf, data_dict, features_list, folds = 1000)


# With features "salary", "bonus", "fraction_from_poi", "fraction_to_poi" I have a precision and recall score both being .31 which is acceptable.

# ## Algorithm Selection

# In[7]:

#going to try Naive Bayes, Decision tree and SVM
from sklearn.cross_validation import StratifiedShuffleSplit

feature_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi"]

data = featureFormat(data_dict, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
cv = StratifiedShuffleSplit(labels, random_state = 42)
for train_idx, test_idx in cv: 
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

print(labels_test)

clf = GaussianNB()
print(test_classifier(clf,data_dict,feature_list))


from sklearn import tree
clf = tree.DecisionTreeClassifier()
print(test_classifier(clf,data_dict,feature_list))


# Of the two algorithms the best one was the Decision Tree Classifier , tuning of algorithm is done below.
# 
# ## Algorithm Tuning

# In[8]:

print('None')
clf = tree.DecisionTreeClassifier()
print(test_classifier(clf,data_dict,feature_list))
print('sqrt')
clf = tree.DecisionTreeClassifier(max_features = 'sqrt')
print(test_classifier(clf,data_dict,feature_list))
print('log 2')
clf = tree.DecisionTreeClassifier(max_features = 'log2')
print(test_classifier(clf,data_dict,feature_list))
print('split 2')
clf = tree.DecisionTreeClassifier(min_samples_split = 2)
print(test_classifier(clf,data_dict,feature_list))
print('sqrt, split 2')
clf = tree.DecisionTreeClassifier(max_features = 'sqrt',min_samples_split = 5)
print(test_classifier(clf,data_dict,feature_list))
print('log2, split 2')
clf = tree.DecisionTreeClassifier(max_features = 'log2',min_samples_split = 8)
print(test_classifier(clf,data_dict,feature_list))


# Best tuned algorithm has a (max features = sqrt) argument rather than None.

# ## Evaluation and Validation
# 

# In[9]:

clf = tree.DecisionTreeClassifier()
print(test_classifier(clf,data_dict,feature_list))
print('\n')
print('Versus Optimized:')#vs
print('\n')
print('sqrt')
clf = tree.DecisionTreeClassifier(max_features = 'sqrt')
print(test_classifier(clf,data_dict,feature_list))


# Validation uses k=1000 folds. Meaning it uses 10% of the data as a trainer (which is default) and uses 90% for testing. It then averages results over 1000 runs with a differnet 10% training data each time. 

# ## Dumping

# In[10]:

features_list = ["poi", "salary", "bonus", "fraction_from_poi", "fraction_to_poi"]
my_dataset = data_dict
clf = tree.DecisionTreeClassifier(max_features = 'sqrt')
dump_classifier_and_data(clf, my_dataset, features_list)


# Review of tester.py output reveals we achieved sucess! Both precission and recall are above .3!

# In[ ]:



