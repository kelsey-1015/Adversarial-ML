====================RAW_TRACEFILE==============================

All the traces are installed in the raw_tracefile folder
====================RESULTS_DATA===============================
-- RESULTS_DATA_FOR_PLOTTING: json results for plotting Figures in the article
-- results_data_wrapup: intermediate results during the experiments


====================CODE=======================================
contants.py ==> The mapping between app-attack and filenames; The mapping between app-feature and filenames
train_model.py ==> Train and Test an OC-SVM model with different settings. Output FPR/TPR
plot.py --> draw plots
fpr_reduction_analysis.py ==> Perform the anomaly analysis module and plot Figure 4
oc_svm.py ==> Conduct the training of oc-svms
trace_file_parser.py ==> parse the raw traces and extract the keys of the [item- pair] dictionary for different feature
vectors, the output are saved in folder "feature_vector_json"
====
feature_vector_json ==> contains json files containing the list of unique syscall symbol with different applications
and feature exaction methods
