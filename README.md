# Data Science Helpers
This package provides a variety of useful functions for data science for exploratory analysis, correlation analysis, 
classification, regression, etc. It contains a handful of useful utility functions as well as modern-looking and 
easy-to-use visualizations that make playing with the data a real joy.

The main package that contains all the magic is named **helpers**. For more information, please see the 
description provided in the docstrings or take a look at the **jupyter notebooks**  for each of the packages.

This package is developed by the members of [Brain Disease Analysis Laboratory](http://bdalab.utko.feec.vutbr.cz/) 
(**BDALab**). For more information, please contact the main author: Zoltan Galaz at <z.galaz@feec.vutbr.cz>.

* * * * * * * * *
 
## Installation
```
git clone git@github.com:zgalaz/data-science-helpers.git
cd data-science-helpers
python3 -m virtualenv .venv
source .venv/bin/activate
pip install .
```

## Structure
```
+---helpers
|   +---classification
|   |   |   metrics.py
|   |   |   validation.py
|   |   |   visualization.py
|   |           
|   +---common
|   |   |   visualization.py
|   |           
|   +---correlation
|   |   |   computation.py
|   |   |   visualization.py
|   |           
|   +---exploration
|   |   |   visualization.py
|   |           
|   +---regression
|   |   |   metrics.py
|   |   |   validation.py
|   |           
|   \---utils
|       |   logger.py
|       |   validators.py
|               
+---notebooks
    |   classification_notebook.ipynb
    |   correlation_notebook.ipynb
    |   exploration_notebook.ipynb
    |   regression_notebook.ipynb
```

# License
This project is licensed under the terms of the MIT license. For more details, see the **LICENSE** file.