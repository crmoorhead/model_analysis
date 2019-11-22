# model_analysis

Tools for comparing the performance of a model using the test set, both statistically and visually.

These include:

- __test_model(ioTT,model,source_dir,\*args,\*\*kwargs)__: Generated predictions on the test set using a given model and compares with the ground truth for each input. ioTT is of the format produced by the input\_output function (see the documentation for dl\_pipeline). model is a keras model object trained on the training set, although this argument accepts lists of such objects in order to process multiple models at once and a string or list of strings corresponding to the path or paths of models previously trained on the same data. The source\_dir is the associated source directory containing the test data. There are two optional arguments, "greyscale" which will convert the loaded images to single layer array and "print_test_results" which will print the test accuracy to screen. There are also two optional keyword arguments. The first is "exceptions_dir" which must have a value of a string name and will be a directory created within the source_dir where misclassifications will be saved 

The output to this model forms the basis for any further visualisations of the manner in which inputs are classified or misclassified. 

- __change_labels(io,"args")__: Takes a dictionary of the format of the input_output function and returns a dictionary mapping integer labels to class descriptions. There is only one optional argument "

- __plot\_confusion\_matrix__(y_true,y_pred,classes,\*args, \*\*kwargs): Creates a Confusion Matrix plot using the output to the test\_model function above. There should also be an argument for changing the class labels from integers to whatever their original description was. 


- __Precision and Recall__
- __Confusion Matrix__
- __AUC plot__

