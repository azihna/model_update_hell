
# Updating the model in production: The Horror Show

This is the source code created initially for PyCon Ireland '23. Here's the abstract for the talk:
You have built your data pipeline, your model is trained, stakeholders are happy, and the model is in production. But the hard questions don't stop there. How and when to update your model?
In this talk, I'll share my experience in maintaining production models, introduce the tools we use in CKDelta for this purpose and approaches we use to diagnose decaying model performance and replace them when necessary.
Familiarity with data pipelines, machine learning models and common machine learning frameworks is required.


## Installation

Install the environment with conda

```bash
  conda env create -f environment.yml
  conda activate update_hell
```

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
