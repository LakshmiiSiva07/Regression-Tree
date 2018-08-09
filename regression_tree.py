import numpy as np
import csv
import random
import math

import matplotlib.pyplot as plt


ratio = 0.5
max_split_points = 75
train_test_split = 320
no_of_bag_learners = 100
min_no_of_tuples_in_a_leaf_node = 20
file = open('Carseats_preprocessed.csv', "rb")
Feautures = [[0,'Sales'],[1,'CompPrice'],[2,'Income'],[3,'Advertising'],[4,'Population'],[5,'Price'],[6,'ShelveLoc'],[7,'Age'],[8,'Education'],[9,'Urban'],[10,'US']]
Total_splits = 0
alpha = 0.0
k_folds = 10


#Loading data from a csv file
def load_data():
    print ("Loading Data\n")
    lines = csv.reader(file)
    dataset = list(lines)
    return dataset

#Convert String datatype to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
      #  row[column] = int(row[column])

 #Returns no of leaf nodes in a tree
def no_of_leaf_nodes():
   x = Total_splits + 1
   return x

#Splitting a node
def splitset (parentnode,split_index,split_point):
    right_child_list = []
    left_child_list = []
    for i in range(len(parentnode.root)):
        if (parentnode.root[i][split_index] >= split_point):
            right_child_list.append(parentnode.root[i])
        else:
            left_child_list.append(parentnode.root[i])
    parentnode.left_node = Node(left_child_list)
    parentnode.right_node = Node(right_child_list)
    parentnode.leaf = 0
    parentnode.split_point = split_point
    parentnode.feature = split_index

#Node data structure
class Node(object):
    def __init__(self,rootvalue):
        self.root = rootvalue
        self.left_node = None
        self.right_node = None
        self.feature = 0
        self.split_point = 0
        self.mean = self._set_mean()
        self.leaf = 1

    #Calculates the mean value for y predicate
    def _set_mean(self):
        sum = 0
        for i in range(len(self.root)):
            sum = sum + self.root[i][0]
        #print len(self.root)
        return sum/len(self.root)


    def clone(self):
        p = Node(self.root)
        p.left_node = self.left_node
        p.right_node = self.right_node
        p.split_point = self.split_point
        p.feature = self.feature
        p.mean = self.mean
        p.leaf = self.leaf
        return p

     # Calculates residual sum square error
    def residual_sum_square(self):
        error = 0
        for i in range(len(self.root)):
            error = error + ((self.root[i][0] - self.mean) ** 2 )
        #print error
        return error

#generates_random_split_points
def generate_split_points(node):
    split_array = []
    for i in range(1,len(node.root[0])):
        feauture_points = []
        for j in range(len(node.root)):
            feauture_points.append(node.root[j][i])
        for j in range(max_split_points):
            split_array_per_feature = []
            split_array_per_feature.append(i)
            if (min(feauture_points) != max(feauture_points)):
                split_array_per_feature.append(random.randint((min(feauture_points)+1),(max(feauture_points))))
                split_array.append(split_array_per_feature)
    return split_array

def generate_split_points_random(node):
    split_array = []
    num_features = int(math.ceil(math.sqrt(10))) + 1
    for i in range(1,num_features):
        feature_index = random.randint(1,(len(node.root[0]) - 1))
        feauture_points = []
        for j in range(len(node.root)):
            feauture_points.append(node.root[j][feature_index])
        for j in range(max_split_points):
            split_array_per_feature = []
            if (min(feauture_points) != max(feauture_points)):
                split_array_per_feature.append(feature_index)
                split_array_per_feature.append(random.randint((min(feauture_points)+1),(max(feauture_points))))
                split_array.append(split_array_per_feature)
    #print split_array
    return split_array

#Gives the best feauture to be split and best split index
def generate_best_feature(parentnode):

    split_list = generate_split_points(parentnode)
    rss_list = []
    rss_min = 99999
    for i in range (len(split_list)):
       node = parentnode.clone()
       #print  split_list[i][0],split_list[i][1]
       splitset(node,split_list[i][0],split_list[i][1])
       rss = node.right_node.residual_sum_square() + node.left_node.residual_sum_square()
       rss = rss + alpha*no_of_leaf_nodes()
       if rss_min > rss:
           rss_min = rss
           x = (node.feature,node.split_point,rss_min)
    rss_list.append(x)
    return x

#Random Forest
def generate_best_feature_random(parentnode):
    split_list = generate_split_points_random(parentnode)
    rss_list = []
    rss_min = 99999
    x = []
    for i in range (len(split_list)):
       node = parentnode.clone()
       #print  split_list[i][0],split_list[i][1]
       splitset(node,split_list[i][0],split_list[i][1])
       rss = node.right_node.residual_sum_square() + node.left_node.residual_sum_square()
       rss = rss + alpha*no_of_leaf_nodes()
       if rss_min > rss:
           rss_min = rss
           x = (node.feature,node.split_point,rss_min)
    rss_list.append(x)
    return x


#Builds decision tree
def makeregressiontree(parentnode):
    global Total_splits
    #print "Making tree"
    if (len(parentnode.root) > min_no_of_tuples_in_a_leaf_node):
        split_index,split_point,rss_child = generate_best_feature(parentnode)
        #print split_index,split_point
        rss_parent = parentnode.residual_sum_square()
        if (rss_parent > rss_child):
            Total_splits = Total_splits + 1
            splitset(parentnode,split_index,split_point)
            makeregressiontree(parentnode.left_node)
            makeregressiontree(parentnode.right_node)
        return parentnode
    else:
        return parentnode

#Make Tree for random forest
def makerandomregressiontree(parentnode):
    global Total_splits
    #print "Making tree"
    if (len(parentnode.root) > min_no_of_tuples_in_a_leaf_node):
        split_index,split_point,rss_child = generate_best_feature_random(parentnode)
        #print split_index,split_point
        rss_parent = parentnode.residual_sum_square()
        if (rss_parent > rss_child):
            Total_splits = Total_splits + 1
            splitset(parentnode,split_index,split_point)
            makerandomregressiontree(parentnode.left_node)
            makerandomregressiontree(parentnode.right_node)
        return parentnode
    else:
        return parentnode

def print_tree(Tree):

    if (Tree.leaf == 0):
        #print len(Tree.root)
        print Feautures[int(Tree.feature)][1],
        print "<",
        print Tree.split_point,
        print_tree(Tree.left_node)
        print Feautures[int(Tree.feature)][1],
        print ">=",
        print Tree.split_point,
        print_tree(Tree.right_node)
    else:
        print round(Tree.mean,2)

#Runs the test
def predict(Final_tree,test_dataset):
    mse = 0
    for i in range(len(test_dataset)):
        predicted_value = predict_yvalue(Final_tree,test_dataset[i])
      #  print "Actual Sales value " + str(test_dataset[i][0]),
       # print "Predicted Sales value " + str(round(predicted_value,2))
        error = ((int(test_dataset[i][0]) - predicted_value) ** 2 )
        mse = mse + error
    return float(mse/float(len(test_dataset)))

#Aggregates all bagged learners prediction
def predict_bagging(Bagging_Learners,test_dataset):

    mse = 0
    for i in range(len(test_dataset)):
        y_value = []
        for learner in Bagging_Learners:
            y_value.append(predict_yvalue(learner,test_dataset[i]))
        #print "Actual Sales value " + str(test_dataset[i][0]),
        #print y_value
        #print np.mean(y_value)
        error = ((int(test_dataset[i][0]) - np.mean(y_value)) ** 2)
        mse = mse + error
    return float(mse / float(len(test_dataset)))

#Predicts sales value based on mean
def predict_yvalue(Node,test_row):
    y = 0
    while (Node.leaf == 0):
        if (test_row[Node.feature] < Node.split_point):
            Node = Node.left_node
        else:
            Node = Node.right_node
    x = Node.mean
    return x

#Cross validation
def cross_validite(dataset):
    global Total_splits,alpha
    dataset_split = []
    dataset_copy = dataset
    alpha_list = []
    mse_list = []
    fold_size = int(len(dataset) / k_folds)
    for i in range(k_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)

    for i in range(10) :
        alpha_list.append(random.uniform(0,10))
    #print alpha_list
    print "MSE  ALHA  Total_splits"
    for i in range(10):
        MSE_LIST = []
        TSP_AVG = []
        alpha = alpha_list[i]
        for  i in range(len(dataset_split)):
            train_set_cv = []
            train_set = list(dataset_split)
            test_set =  train_set.pop(i)
            for row in train_set:
                for column in row:
                    train_set_cv.append(column)
            Total_splits = 0
            Tree = makeregressiontree(Node(train_set_cv).clone())
            MSE = predict(Tree,test_set)
            MSE_LIST.append(MSE)
            TSP_AVG.append(Total_splits)
           # print MSE,
           # print Total_splits,
            #print alpha
        print round(np.mean(MSE_LIST),2),round(alpha,2),np.mean(TSP_AVG)
        mse_list.append(np.mean(MSE_LIST))
   # plt.plot(mse_list,alpha_list)
    #plt.show()

#Returns sub-samples for bagging based on ratio
def subsample(dataset, ratio):
    sample = []
    dl = list(dataset)
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = random.randrange(len(dl))
        x = dl.pop(index)
        sample.append(x)
    return sample

#Performs learning using bagging method
def bagging(dataset,test_dataset):
     Bagging_Learner = []
     for i in range(no_of_bag_learners):
        print 'Learner',
        print i
        sample = subsample(dataset,ratio)
       # print sample
        Bagging_Learner.append(makeregressiontree(Node(sample).clone()))
        importance(Bagging_Learner[i])
     MSE = predict_bagging(Bagging_Learner,test_dataset)
     print 'MSE:'
     print MSE

#Generates Random Forest
def random_forest(dataset,test_dataset):
    Bagging_Learner = []
    for i in range(no_of_bag_learners):
        print 'Learner',
        print i
        sample = subsample(dataset, ratio)
        # print sample
        Bagging_Learner.append(makerandomregressiontree(Node(sample).clone()))
        importance(Bagging_Learner[i])
    MSE = predict_bagging(Bagging_Learner, test_dataset)
    print 'MSE:'
    print MSE

#List the three features responsible for top-splits
def importance( regression_tree):
      if (regression_tree.leaf == 0):
        print  Feautures[regression_tree.feature][1],
        print  Feautures[regression_tree.left_node.feature][1],
        print  Feautures[regression_tree.right_node.feature][1]

def main():

    #random.seed(30)
    dataset = load_data()

    print("Converting into float\n")
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    train_dataset = dataset[:train_test_split]       #Splitting dataset into train and test set
    test_dataset = dataset[train_test_split:]

    Decisiontree = Node(train_dataset)
    Tree = Decisiontree.clone()
    Final_tree = makeregressiontree(Tree)
    print_tree(Final_tree)
    MSE = predict(Final_tree,test_dataset)
    print MSE
    print "Total_splits",
    print Total_splits
    print "Important Feautures:"
    importance(Final_tree),
    print "Cross validating"
    cross_validite(dataset)
    print "Bagging"
    bagging(train_dataset,test_dataset)
    print "Random forests"
    random_forest(train_dataset,test_dataset)


if __name__ == "__main__":
    main()

