# Introduction

Machine learning project where the main object is to implement algorithms, understand them, evaluate results and compare them with other research papers and their results.

Dataset used for this study is [Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset). [[1]](#1)

# Preparations

## Prerequisite

Follow this steps:

 * Install python. The one currently used is 3.11. A newer version will work as well but take care of the requirements dependecies from [requirements.txt](requirements.txt) - it may not work with current versions from the file. Newer packages version may be available. Use them at your own risk.

 * Create a python environment, with your available python, in the project folder.
`python3 venv .venv`.

 * Activate the created environment by writing `source .venv/bin/activate`.

 * Install project requirements by running `python -m pip install -r requirements.txt`. This may take a while.

_**Note:** All this commands are executed on Unix/Linux system. For Windows, it may be required to adjust slightly._

## Usage

The data is stored in arff, txt and xlsx format in [data](/data/) folder. [[1]](#1)

If there is a need to import data direct in python without usage of [data](/data/) folder run [00_import_dataset.ipynb](/notebooks/00_import_dataset.ipynb) first.

The entire implementation is made in python and usage of jupyter notebooks. All of them can be found in [notebooks](/notebooks/). Start with [01_eda.ipynb](/notebooks/01_eda.ipynb) and follow the rest of them.

# Execution

## Scope

Implement three models:
 * Regularized Logistic Regression 
 * Neural Network Classification (NNC)
 * Decision Trees

Compare results with reference papaers.

## Reference

<a id="1">[1]</a>
Dry Bean [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C50S4B.

<a id="2">[2]</a>
Koklu, Murat & Ozkan, Ilker Ali. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. Computers and Electronics in Agriculture. 174. 105507. 10.1016/j.compag.2020.105507.

<a id="3">[3]</a>
Shobana, G. et al. “Multivariate Classification of Dry Beans using Pipelined Dimensionality Reduction Technique.” 2022 International Conference on Innovative Computing, Intelligent Communication and Smart Electrical Systems (ICSES) (2022): 1-6.

<a id="4">[4]</a>
Yasar, Ali. “Identification of Dry Bean Seeds Using PSO Feature Selection Technique.” 2024 59th International Scientific Conference on Information, Communication and Energy Systems and Technologies (ICEST) (2024): 1-4.

<a id="5">[5]</a>
Krishnan, S. et al. “Identification of Dry Bean Varieties Based on Multiple Attributes Using CatBoost Machine Learning Algorithm.” Scientific Programming (2023): n. pag.

<a id="6">[6]</a>
Shriya, Sakshi et al. “Dry Beans Classification using Ensemble Learning.” 2023 3rd International Conference on Smart Data Intelligence (ICSMDI) (2023): 327-334.

<a id="7">[7]</a>
Kini M G, Ramanath and Rekha Bhandarkar. “Quality Assessment of Seed Using Supervised Machine Learning Technique.” Journal of The Institution of Engineers (India): Series B 104 (2023): 901 - 909.

<a id="8">[8]</a>
Subbarao, M. Venkata et al. “Performance Analysis of Feature Selection Algorithms in the Classification of Dry Beans using KNN and Neural Networks.” 2023 International Conference on Sustainable Computing and Data Communication Systems (ICSCDS) (2023): 539-545.
