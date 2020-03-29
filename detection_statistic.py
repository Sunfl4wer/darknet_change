import os
import glob
import matplotlib.pyplot as plt
import numpy as np

path_to_label_file = "data/objects/" # path to your images and label folder
true_positive_stat_file = "true_positive_stat.txt" # text file that return the true positive anchors
obj_names = "obj.names" # label file 

def readFile(filename):
    true_positive = []
    label_file = []
    with open(filename,'r') as f:
        content = f.readlines()
        true_positive = get_true_positive_bounding_box_stat(content)
        label_file = get_valid_file(content)
    
    return true_positive, label_file

def get_true_positive_bounding_box_stat(content):
    labelfile = []
    number_of_file = 0
    for line in content:
        label = []
        if "TP box" in line:
            label.append(number_of_file)
            word = ""
            for char in line:
                #print(char)
                if char != ' ' and char != '\n':
                    word = word+char
                else:
                    if word != "TP" and word != "box:" and word != "":
                        if '.' not in word:
                            label.append(int(word))
                        else:
                            label.append(float(word))
                    word = ""
        else:
            number_of_file+=1
                        
        if not label:
            continue
        else:
            labelfile.append(label)
    
    return labelfile

def get_valid_file(content):
    valid_file_names = []
    for line in content:
        label = []
        if "Label path:" in line:
            word = ""
            for char in line:
                #print(char)
                if char != ' ' and char != '\n' and char != '/':
                    #print(word)
                    word = word+char
                else:
                    if word != "Label" and word != "" and word != "path:" and word != "data" and word != "objects":
                        #print(word)
                        label.append(word)
                    word = ""
                        
        if not label:
            continue
        else:
            valid_file_names.append(label)
    
    return valid_file_names

def get_false_negative_bounding_box_stat(file_name, true_positive):
    false_negative = []
    number_of_file = 0
    for i in range(len(file_name)):
        tp_in_file = []
        labels_in_file = []
        fp_in_file = []
        name = file_name[i][0]
        for tp_label in true_positive:
            if tp_label[0]-1 == i:
                tp_in_file.append(tp_label)
            elif tp_label[0]-1 > i:
                break
        #print(file)
        number_of_file+=1
        with open(path_to_label_file+name,'r') as f:
            content = f.readlines()
            for line in content:
                label = []
                label.append(number_of_file)
                word = ""
                for char in line:
                    #print(char)
                    if char != ' ' and char != '\n':
                        word = word+char
                    else:
                        if '.' not in word:
                            label.append(int(word))
                        else:
                            label.append(float(word))
                        word = ""
                labels_in_file.append(label)
        fp_in_file = [x for x in labels_in_file if x not in tp_in_file]
        for fp in fp_in_file:
            false_negative.append(fp)    
    return false_negative

def plot_data(in_data, classes):
    no_class = len(classes)
    width = []
    height = []
    
    for i in range(no_class):
        line_width = []
        line_height = []
        for line in in_data:
            if line[1] == i:
                line_width.append(line[4])
                line_height.append(line[5])
        width.append(line_width)
        height.append(line_height)
    
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    no_col = 3
    for i in range(no_class):
        if no_class % no_col == 0:
            subplot = no_class/no_col*100+no_col*10+i+1
        else:
            subplot = (no_class+1)/no_col*100+no_col*10+i+1
        ax=fig.add_subplot(subplot)
        ax.scatter(width[i], height[i], color='r', label=classes[i])
        ax.legend(fontsize='small')
    plt.show()

def get_classes(file):
    classes = []
    with open(file,'r') as f:
        content = f.readlines()
        for line in content:
            if not line:
                continue
            else:
                classes.append(line)
    return classes

def count_number_of_instance(list_of_label, classes):
    count = [0] * len(classes)
    for i in range(len(classes)):
        for label in list_of_label:
            if label[1] == i:
                count[i] += 1
    return count
    
def calcualte_recall(true_positive_count, false_negative_count):
    recall_result = [0] * len(true_positive_count)
    for i in range(len(true_positive_count)):
        recall_result[i] = true_positive_count[i]/(true_positive_count[i]+false_negative_count[i])*100
    
    return recall_result

def plot_recall_bar(data, labels):
    y_pos = np.arange(len(labels))
    plt.bar(y_pos,data)
    plt.xticks(y_pos, labels, fontsize = 'small')
    plt.ylabel("%")
    plt.title("Recall")
    plt.show()

true_positive, file_name = readFile(true_positive_stat_file)
print("True positive count: %d " %(len(true_positive)))
false_negative = get_false_negative_bounding_box_stat(file_name, true_positive)
print("False negative count: %d " %(len(false_negative)))
classes = get_classes(obj_names)
plot_data(true_positive, classes)
plot_data(false_negative, classes)
#no_tp_each_class = count_number_of_instance(true_positive,classes)
#no_fn_each_class = count_number_of_instance(false_negative,classes)
#recall_result = calcualte_recall(no_tp_each_class, no_fn_each_class)
#print(recall_result)
#plot_recall_bar(recall_result, classes)
