
# Updating the model in production: The Horror Show

This is the source code created initially for PyCon Ireland '23. Here's the abstract for the talk:
You have built your data pipeline, your model is trained, stakeholders are happy, and the model is in production. But the hard questions don't stop there. How and when to update your model?
In this talk, I'll share my experience in maintaining production models, introduce the tools we use in CKDelta for this purpose and approaches we use to diagnose decaying model performance and replace them when necessary.
Familiarity with data pipelines, machine learning models and common machine learning frameworks is required.


## Installation

### Using Conda

```bash
  conda env create -f environment.yml
  conda activate update_hell
```

### Using pip

Use with a virtual environment.

```bash
  pip install -r requirements.txt
```

### Using Docker

You might need to change the ip address or the ports based on the available ports on local.

```bash
  docker image built -t <image_name> .
  docker run -it -p 8888:8888 <image_name>
```

I generally use VSCode but to use default Jupyter Notebook or Lab.
Run following to have access in local.

```bash
  conda activate update_hell
  jupyter notebook --ip 0.0.0.0 --allow-root
```

Copy the shown links to access the enviroments.

## Structure

├── data  
│   ├── full_data  # joined full data  
│   ├── initial  # location of the historical database  
│   ├── raw  # Downloaded kaggle data as-is  
│   └── updates  # Daily update location  
├── solutions  # folder with example updates  
├── data_prep.py  # script used to create the files  
├── environment.yml  # conda env file  
└── train_and_predict.py  # starter location for the presentation  

## Dataset

Dataset is based on [Wholesale & Retail Orders](https://www.kaggle.com/datasets/gabrielsantello/wholesale-and-retail-orders-dataset).

Changes to the original data:
* Column names normalized
* Removed the same day deliveries
* Merged two datasets
