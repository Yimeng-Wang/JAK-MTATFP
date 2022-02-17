# JAK-MTATFP

A selective JAK inhibitor screening and designing platform. 
## Installation

The code in this repository relies on the DGL package (https://github.com/dmlc/dgl) with pytorch backend (http://pytorch.org) as well as on sklearn (http://scikit-learn.org) and dgllife (https://github.com/awslabs/dgl-lifesci).
We are recommended you to create a conda environment for example:

`conda env create -f environment.yml`  

Then activate the environment:  
  
`conda activate MTATFP`

## Usage

First of all, we are required to prepare the molecules, taking 'C/C=C/C(O)=Nc1cccc(CNc2c(C(=N)O)cnn3cccc23)c1' as an exapmle:  
  
`python preprocess.py`
    
And then, you can train the model, using the following command：
  
`python MTATFP_train.py`
  
Or, you could directly use our trained model with the preprocessed molecules to make predictions:  
  
  `python MTATFP_test.py`  
  
The trained STATFP models were also provided here, you can also used them by the **STATFP_test.py**. And The LightGBM models for machine learning methods are also provided in repository.   
You could open the **atom_visualization.ipynb** in Jupyter Notebooks to visualise atom features. 
## Cite
