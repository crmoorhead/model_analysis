# model_analysis

import cv2
from dl_pipeline import *

from keras.utils.vis_utils import plot_model

def model_info(model,*args,**kwargs):
    if "summary" in args:
        print(model.summary())
    if "layer_count" in args:
        num_of_layers=len(model.layers)
        print("Distinct named layers in",model.name,":",num_of_layers)
    if "layer_names" in args:
        print("Layer names for model:",model.name)
        print()
        layer_names = [layer.name for layer in model.layers]
        for layer in layer_names:
            print(layer)
    if "diagram" in args:
        if "save_dir" in kwargs:
            save_path=kwargs["save_dir"]+"\\"+model.name+".png"
        else:
            save_path=model.name+".png"
        if "display_tensors" in args:
            if "display_names" in args:
                plot_model(model,to_file=save_path, show_shapes=True, show_layer_names=True)
            else:
                plot_model(model, to_file=save_path + ".png", show_shapes=True, show_layer_names=False)

        else:
            if "display_names" in args:
                plot_model(model, to_file=save_path + ".png", show_shapes=False, show_layer_names=True)
            else:
                plot_model(model, to_file=save_path + ".png", show_shapes=False, show_layer_names=False)
    if "save_summaries" in args:
        if "save_dir" in kwargs:
            save_path=kwargs["save_dir"]+"\\"+model.name+".txt"
        else:
            save_path =model.name + ".txt"
        f=open(save_path,"w")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.close()

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

# TEST DATA

'''xs1=[0,1,2,3,4,5,6]
xs2=[0,1,2,3,4,5,6]

number_of_trials=100
ys1=[[0.5+0.1*j+normalvariate(0,0.05) for j in range(7)] for i in range(number_of_trials)]
ys2=[[0.2+0.2*j+normalvariate(0,0.07) for j in range(7)] for i in range(number_of_trials)]'''

# FUNCTION TO PLOT MANY RUNS OF A GIVEN EXPERIMENT AND COMPARE ON SAME AXIS

# FUNCTION TO PREPARE DATA

# The input must be a list of lists. First entry must be the x-coordinates of the plot while the second is a list of lists of
# y-coordinates for each run of the experiment. The output of the function will be a dictionary with keywords "x_coords","labels","mean","max","min" and optionally
# "median", "Q1" and "Q3" for the lower and upper quartiles respectively and "std" and "2std".

def create_data_dict(data_runs,*args,**kwargs):
    # check that data is of right form and lengths match
    if data_runs.__class__!=list or data_runs[0].__class__!=list or data_runs[1].__class__!=list:
        print("Check format of input")
        return None
    number_of_xs=len(data_runs[0])
    if data_runs[1][0].__class__!=list:
        number_of_trials=1
        if number_of_xs!=len(data_runs[1]):
            print("Data points for x and y are of different length.")
            return None
        data_runs[1]=[data_runs[1]]
    else:
        number_of_trials=len(data_runs[1])
        if False in [number_of_xs==len(data_runs[1][i]) for i in range(number_of_trials)]:
            print("Data points for x and y are of different length for at least one run.")
            return None

    # Create dictionary
    data_dict={"x_coords":data_runs[0]}
    # Calculate statistical properties and add to data_dict

    data_dict["mean"]=[sum([data_runs[1][i][j] for i in range(number_of_trials)])/number_of_trials for j in range(number_of_xs)]
    data_dict["max"] = [max([data_runs[1][i][j] for i in range(number_of_trials)]) for j in range(number_of_xs)]
    data_dict["min"] = [min([data_runs[1][i][j] for i in range(number_of_trials)]) for j in range(number_of_xs)]

    # Quartiles

    data_dict["median"] = [np.percentile([data_runs[1][i][j] for i in range(number_of_trials)],50) for j in range(number_of_xs)]
    data_dict["Q1"] = [np.percentile([data_runs[1][i][j] for i in range(number_of_trials)],25) for j in range(number_of_xs)]
    data_dict["Q3"] = [np.percentile([data_runs[1][i][j] for i in range(number_of_trials)],75) for j in range(number_of_xs)]

    sds = [np.std([data_runs[1][i][j] for i in range(number_of_trials)]) for j in range(number_of_xs)]
    data_dict["+1 SD"]=[data_dict["mean"][i]+sds[i] for i in range(number_of_xs)]
    data_dict["-1 SD"] = [data_dict["mean"][i] - sds[i] for i in range(number_of_xs)]
    data_dict["+2 SD"] = [data_dict["mean"][i] + 2*sds[i] for i in range(number_of_xs)]
    data_dict["-2 SD"] = [data_dict["mean"][i] - 2*sds[i] for i in range(number_of_xs)]

    if "exp_name" in kwargs:        # Sets the name of set of experiments used to plot a single line
        data_dict["name"]=kwargs["exp_name"]
    else:
        data_dict["name"]="Experimental results"

    if "labels" in kwargs:
        data_dict["x_label"]=kwargs["labels"][0]
        data_dict["y_label"] = kwargs["labels"][1]
    else:
        data_dict["x_label"]="Independent variable"
        data_dict["y_label"] = "Dependent variable"

    return data_dict


test_data1=create_data_dict([xs1,ys1],exp_name="Experiment 1")
test_data2=create_data_dict([xs2,ys2],exp_name="Experiment 2")

# Function to create plot

def multi_model_stats(data_stats,*args,**kwargs):
    if "alpha" in kwargs:
        alpha=kwargs["alpha"]
    else:
        alpha=0.5
    if "colors" in kwargs:
        colors=kwargs["colors"]
    else:
        colors=["green","blue","red","yellow","purple","pink","grey"] # CHECK THESE ARE ALL VALID!
        if len(data_stats)>len(colors):
            print("You need to define a colour scheme for more than 7 sets of experiments.")
        i=0
        for d in data_stats:
            d["color"]=colors[i]
            i+=1
    if "title" in kwargs:
        chart_title=kwargs["title"]
    else:
        chart_title=""
    fig, ax = plt.subplots(1) # ready workspace
    for d in data_stats:
        ax.plot(d["x_coords"], d["mean"], lw=2, label=d["name"], color=d["color"]) # Default line width of 2
        ax.fill_between(d["x_coords"], d["max"], d["min"], facecolor=d["color"], alpha=alpha)
    if "percentiles" in args:
        for d in data_stats:
            ax.plot(d["x_coords"], d["Q1"], lw=1, ls=":",color=d["color"]) # Default line width of 1
            ax.plot(d["x_coords"], d["Q3"], lw=1, ls=":",color=d["color"])  # Default line width of 1
    if "median" in args:
        for d in data_stats:
            ax.plot(d["x_coords"], d["median"], lw=1, label=d["name"], color=d["color"]) # Default line width of 1
    if "std" in args:
        for d in data_stats:
            ax.plot(d["x_coords"], d["+1 SD"], lw=1, ls="--",color=d["color"])  # Default line width of 1
            ax.plot(d["x_coords"], d["-1 SD"], lw=1, ls="--",color=d["color"])  # Default line width of 1
    if "2std" in args:
        for d in data_stats:
            ax.plot(d["x_coords"], d["+2 SD"], lw=1, ls="--", color=d["color"])  # Default line width of 1
            ax.plot(d["x_coords"], d["-2 SD"], lw=1, ls="--", color=d["color"])  # Default line width of 1
    ax.set_title(chart_title)
    if len(data_stats)>1:
        ax.legend(loc='upper left')
    ax.set_xlabel(choice(data_stats)["x_label"])
    ax.set_ylabel(choice(data_stats)["y_label"])
    if "grid" in args:
        ax.grid()
    if "rescale" in args:
        # Put options for scaling X and Y axes
        pass
    plt.show()

# multi_model_stats([test_data1,test_data2],"2std","std",title="Experiment under different settings")

