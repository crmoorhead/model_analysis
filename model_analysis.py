# model_analysis

import cv2
from dl_pipeline import *

def test_model(ioTT,model,source_dir,*args,**kwargs):
    if str(model.__class__)=="<class 'keras.engine.training.Model'>":   # If model input is a Keras Model, we proceed as normal.
        if "exceptions_dir" in kwargs:
            from os import mkdir,listdir
            from shutil import copy
            if "test_misclassifications" not in listdir(kwargs["exceptions_dir"]):
                mkdir(kwargs["exceptions_dir"]+"\\"+"test_misclassifications")
        score=0
        pred_real={}
        for io in ioTT:
            if "greyscale" in args:
                input_array=np.expand_dims(np.array([np.array(cv2.imread(source_dir+"\\"+io))[:,:,0]]),-1)
            else:
                input_array=np.array(cv2.imread(source_dir+"\\"+io))
            if "augmentations" in kwargs:   # For applying augmentations to the test set
                input_array = apply_augment(input_array, kwargs["augmentations"], image_functional, *args, **kwargs)
            input_array=input_array/255
            vector=list(model.predict(input_array)[0])
            class_guess=vector.index(max(vector))
            true_class=list(ioTT[io]).index(1)
            pred_real[io]=[class_guess,true_class]
            if class_guess!=true_class:
                if "exceptions_dir" in kwargs:
                    copy(source_dir+"\\"+io,kwargs["exceptions_dir"]+"\\"+"test_misclassifications"+"\\"+str(io[:-4])+" T="+str(true_class)+" G="+str(class_guess)+".jpg")
            else:
                score+=1
        test_dict={"test_acc": score / len(ioTT), "comparisons":pred_real}  # We return the accuracy
        if "print_test_results" in args:
            print(model.name," accuracy on Test Set:",test_dict["test_acc"])
        return test_dict
    elif model.__class__==str:             # If we give the filename of the model input
        model=load_model(source_dir, model,*args,**kwargs)
        test_model(ioTT,model,source_dir,*args,**kwargs)
    elif model.__class__==list:
        if "exceptions_dir" in kwargs:
            pattern=kwargs["exceptions_dir"]
        all_models_test={}
        for m in model:
            if m.__class__==str:
                if "exceptions_dir" in kwargs:
                    kwargs["exceptions_dir"]=pattern+"("+m+")"
                all_models_test[m]=test_model(ioTT,m,source_dir,*args,**kwargs)
            else:
                if "exceptions_dir" in kwargs:
                    kwargs["exceptions_dir"]=pattern+"("+m.name+")"
                all_models_test[m.name]=test_model(ioTT,m,source_dir,*args,**kwargs)
        return all_models_test
    else:
        print("model is not of a compatible class")



# From confusion matrix code

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes,*args,**kwargs):
    if "title" in kwargs:
        title=kwargs["title"]
    else:
        title=""
    cm = confusion_matrix(y_true, y_pred)
    if "cmap" in kwargs:
        cmap_dict={}
        cmap=plt.cm.kwargs["cmap"]
    else:
        cmap = plt.cm.Blues

    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]  # The original list of classes may be more than what is in the data. Is this a necessary step?

    if "normalize" in args or "normalise" in args:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

