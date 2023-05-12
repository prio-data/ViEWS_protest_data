# ViEWS_protest_data
Repository for the "Predicting Armed Conflict with Protest Data" project.

The repository contains a query-set notebook including a complete set of calls to fetch and transform the features directly from the ViEWS database. While access to the database is not publicly available, we store the transformed features for each model as a list of dictionaries.
A separate Jupyter notebook contains the main analysis and is divided into three parts including 1) a specification of additional transformations implemented outside of the ViEWS query-manager as well as the definition of each model, 2) the training and prediction of each specified model objects and 3) the evaluation of the predictions which allows to replicate each figure presented in this study. Note that the default setting of the notebook runs the analysis for conflict incidence. To receive the results for conflict onset and the robustness tests, the user should set run\_outcome equals to 'onset', 'onset\_np' or 'incidence\_np' accordingly. The analysis partly relies on functions which are defined in separate python files. A list naming the features included in M0 - M9 is available as YAML file.

To install viewser, please follow the instructions on https://github.com/prio-data/viewser/wiki/. 
