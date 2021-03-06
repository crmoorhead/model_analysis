# model_analysis

Tools for understanding models and comparing the performance of a model using the test set, both statistically and visually.

These include:

- __model_info(model,\*args,\*\*kwargs)__: Function for displaying model summary to understand structure and tensor operations of a model constructed using the network_structure functions. A graph model of the connections and layers in a Keras model can also be saved as an image.

- __create_data_dict(data_runs,\*args,\*\*kwargs)__: Function that prepares a summary of the results of many runs of the same experiment. the input must be a list with the first element being a list of the values of the x_coordinates used in all experiments. The second element is a list or list of lists corresponding to the matching y-coordinates for each run of the experiment. The output is a dictionary containing the statistical properties of all runs of the experiment. The optional keyword arguments are "exp_name" where we can assign a label to the experiment we are running and "labels" where the assigned value must be a two-element list or tuple with the first element being the label for the x-axis and the second being the label for the y-axis. The output contains the following keywords:
"x_coords","mean","max","min","median","Q1","Q3","+1 SD","-1 SD","+2 SD","-2 SD", "name", "x_label" and "y_label" which are mostly intuitive in definition. If the optional keywords are not supplied, default names will be set.

- __multi_model_stats(data_stats,\*args,\*\*kwargs)__: This takes a list of dictionaries corresponding to multiple series of experiments and each dictionary being generated by the previous function. Optional keyword arguments set the parameters of the plot display. 
  - "alpha" sets the transparency of the bands around the mean results for each x-value and must have value between 0 and 1. Default is   0.5 
  - "colors" must be a list of string colour descriptions recognised by matplotlib. If not defined, the function can assign colours automatically for up to 7 different sets of experiments to be plotted on the same axes. 
  - "title" must be a string that will be the title of the plot.

  The function also accepts a number of optional arguments:
  - "percentiles" will add a line for the lower and upper quartile values for each x-axis point.
  - "median" will add a line for median values for each x-axis point.
  - "std" will add lines for one standard deviation above and below the mean for each x-axis point.
  - "2std" will add lines for two standard deviations above and below the mean for each x-axis point.
  - "grid" will add gridlines to the plot.
  - "rescale" will rescale the x and y axis ranges to the data or specified ranges.

- __test_model(ioTT,model,source_dir,\*args,\*\*kwargs)__: Generated predictions on the test set using a given model and compares with the ground truth for each input. ioTT is of the format produced by the input\_output function (see the documentation for dl\_pipeline). model is a keras model object trained on the training set, although this argument accepts lists of such objects in order to process multiple models at once and a string or list of strings corresponding to the path or paths of models previously trained on the same data. The source\_dir is the associated source directory containing the test data. There are two optional arguments, "greyscale" which will convert the loaded images to single layer array and "print_test_results" which will print the test accuracy to screen. There are also two optional keyword arguments. The first is "exceptions_dir" which must have a value of a string name and will be a directory created within the source_dir where misclassifications will be saved 

The output to this model forms the basis for any further visualisations of the manner in which inputs are classified or misclassified. 

- __change_labels(io,"args")__: Takes a dictionary of the format of the input_output function and returns a dictionary mapping integer labels to class descriptions. There is only one optional argument "

- __plot\_confusion\_matrix__(y_true,y_pred,classes,\*args, \*\*kwargs): Creates a Confusion Matrix plot using the output to the test\_model function above. There should also be an argument for changing the class labels from integers to whatever their original description was. 


- __Precision and Recall__
- __Confusion Matrix__
- __AUC plot__

