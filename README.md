# Fin_Lab
Coding for finance

# Setup

## 1. Install Anaconda
#### Windows:
https://docs.anaconda.com/anaconda/install/windows/


#### Mac:
https://docs.anaconda.com/anaconda/install/mac-os/


#### Linux:
https://docs.anaconda.com/anaconda/install/linux/


## 2. Set Up the Virtual Environment via Terminal 

#### Activate environtment:
conda create -n Fin_Lab python=3.6


conda activate

#### Link your environment with Jupyter: 
pip install ipykernel


python -m ipykernel install --user --name=Fin_Lab

#### Packages from github:

git clone https://github.com/piEsposito/blitz-bayesian-deep-learning.git


cd blitz-bayesian-deep-learning


pip install .


## 3. Install Packages

#### Core data analytics

Package Name:

numpy 
- Matrix operations

scipy
- Scientific operations

pandas
- DataFrame operations

statsmodels
- Statistical models

matplotlib
- Data visualization

#### Finance

Package Name:

pandas_datareader
- Reading financial data from the web

yfinance
- yahoo finance web scraper 

arch
- Volatility modelling

cvxopt
- Core optimization

cvxpy
- Wrapper to "nicely" interface into cvxopt

prophet
- Facebook's timeseries forecasting tool

#### Data visualization

Package Name:

seaborn
- Colorful data visualization

bokeh
- Interactive data visualization

geoplotlib
- Create maps and geographical visualization

Altair
- New framework for plotting graphics

plotnine
- Similar to R's ggplot2 using grammer of graphics

#### Machine & deep learning

Package Name:

scikit-learn
- Core machine learning package

xgboost
- The first publicly available gradient boosting package. Released by Tianqi Chen (University of Washington, Seattle)

lightgbm
- Gradient Boosted Decision Trees package (Microsoft)

catboost
- Gradient Boosting Decision Trees package (Yandex)

keras
- High-level neural networks API

Tensorflow
- Deep learning package from Google

#### Natural language processing (NLP)

Package Name:

nltk
- General NLP tasks

textblog
- Creating NLP prototypes quickly

gensim
- NLP applications for topic modelling, document similarity, etc.

scrapy
- Web scraping

spacy
- Production-level NLP library

#### Favourite JupyterLab extensions

See JupyterLab extensions for more details and GitHub for a full list of available extensions for JupyterLab.

Install nodejs

- $ conda install -c conda-forge nodejs

Installing Jupyterlab extensions in Linux as follows:

- $ jupyter labextension install @jupyterlab/<EXT_NAME>
- $ jupyter labextension install jupyterlab_nbmetadata

You can install plugins in Jupyter lab by clicking the jigsaw icon on the menu bar on the menu bar on the right of the editor.


Extension name:

toc
- Table of contents

jupyterlab_variableInspector
- Variable inspector in Jupyter Lab

jupyterlab_nbmetadata
- Allows you to edit the notebook metadata

jupyterlab_go_to_definition
- Jump to definition of a variable or function in JupyterLab notebook and file editor

Installing Jupyter notebook extensions
- $ conda install -c conda-forge jupyter_contrib_nbextensions