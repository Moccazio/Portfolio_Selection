# Fin_Lab

## Setup

## 1. Install Anaconda

#### Windows:
https://docs.anaconda.com/anaconda/install/windows/


#### Mac:
https://docs.anaconda.com/anaconda/install/mac-os/


#### Linux:
https://docs.anaconda.com/anaconda/install/linux/


## 2. Set Up the Virtual Environment via Terminal 

#### Activate environtment and link your environment with Jupyter:

        conda create -n Fin_Lab python=3.6

        conda activate Fin_Lab 

        pip install ipykernel
        
        python -m ipykernel install --user --name=Fin_Lab


## 3. Install Packages

### Core Data Analytics

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

### Data Visualization

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

### Machine & Deep Learning

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

### Natural language processing (NLP)

Loughran-McDonald sentiment word lists to perform sentiment analysis (csv_file)
- https://drive.google.com/file/d/12ECPJMxV2wSalXG8ykMmkpa1fq_ur0Rf/view

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

### Install Packages from GitHub:

- blitz

        git clone https://github.com/piEsposito/blitz-bayesian-deep-learning.git
        cd blitz-bayesian-deep-learning
        pip install .
        
- pyfolio

        git clone https://github.com/quantopian/pyfolio
        cd pyfolio
        pip install .

#### Favourite JupyterLab extensions


- nodejs

        conda install -c conda-forge nodejs


You can install plugins in Jupyter lab by clicking the jigsaw icon on the menu bar on the menu bar on the right of the editor.


- Table of contents

        toc
        
- Variable inspector in Jupyter Lab

        jupyterlab_variableInspector
        
- Allows you to edit the notebook metadata

        jupyterlab_nbmetadata
        
- Jump to definition of a variable or function in JupyterLab notebook and file editor

        jupyterlab_go_to_definition


- Installing Jupyter notebook extensions
        
        conda install -c conda-forge jupyter_contrib_nbextensions
