# model_analysis

Tools for comparing the performance of a model using the test set, both statistically and visually.

These include:

- __test_model(ioTT,model,source_dir,\*args,\*\*kwargs)__: Generated predictions on the test set using a given model and compares with the ground truth for each input. ioTT is of the format produced by the input\_output function (see the documentation for dl\_pipeline). model is a keras model object trained on the training set, although this argument accepts lists of such objects in order to process multiple models at once and a string or list of strings corresponding to the path or paths of models previously trained on the same data. The source\_dir is the associated source directory containing the test data. There are two optional arguments, "greyscale" which will convert the loaded images to single layer array and "print_test_results" which will print the test accuracy to screen. There are also two optional keyword arguments. The first is "exceptions_dir" which must have a value of a string name and will be a directory created within the source_dir where misclassifications will be saved 

The output to this model forms the basis for any further visualisations of the manner in which inputs are classified or misclassified. 

- __plot\_confusion\_matrix__(y_true,y_pred,classes,\*args, \*\*kwargs): Takes compulsory arguments of y\_pred, the true classifications, and y\_pred, the classifications predicted by the model and classes, the classes are the 

- __Precision and Recall__
- __Confusion Matrix__
- __AUC plot__

