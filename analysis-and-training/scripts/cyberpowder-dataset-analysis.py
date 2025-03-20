# + [markdown]
"""
# Dataset Analysis and Model Creation/Training


1. Brief HuggingFace intro; creating an access token; colab notebook login
1. Creating and uploading datasets to HuggingFace
1. Downloading datasets from HuggingFace
1. Visualizing the dataset
1. "Cleaning" the dataset

### Prerequisites

- Join the HuggingFace CyberPowder organization: [link](https://huggingface.co/organizations/cyberpowder/share/VkAxpCJJIebrTqXgdMFxRElyHhnyAJocHQ)
- Review Section VI (AI/ML Workflows) of the [NEU ORAN paper](https://utah.instructure.com/courses/1045795/files/170447527?wrap=1)
"""
# -
# !pip install datasets huggingface_hub plotly plotly_express onnx onnxruntime

# +

# Imports and setup
import pandas as pd
import torch
import numpy as np
import os
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import datasets
import huggingface_hub as hf
import onnx
import onnxruntime as ort
import datetime
import json
import shutil
from huggingface_hub import hf_hub_download
# -


# ## 1. Brief HuggingFace intro; creating an access token; colab notebook login

# ### HuggingFace Introduction
#

# ### Creating an Access Token
# 1. Go to the HuggingFace website: [link](https://huggingface.co/)
# 1. Log in to your account
# 1. Go to your profile settings
# 1. Create a new access token with write access
# 1. Copy the token and save it in a
#    secure location in case you lose track of it later


# log in to huggingface
# (you'll need to enter the access token you created)
hf.login()

username = hf.whoami()['name']
print(f"Logged in as {username}")

# From here on out, calls to the HuggingFace API should be automatically authenticated with your access token.


# ## 2. Creating and Uploading Datasets to Hugging Face
#
# Let's create some random data to use as a dataset
# We'll use numpy to generate random features and targets and throw them into a pandas DataFrame
random_features = np.random.rand(100, 2)  # 100 samples, 2 features
random_targets = np.random.rand(100, 1)  # 100 samples, 1 target
df = pd.DataFrame(random_features, columns=["feature1", "feature2"])
df["target"] = random_targets
df


# Now we'll use the datasets library to create a dataset from the pandas DataFrame and add some metadata
# You could also create a dataset from a csv file, json file, etc.
dataset = datasets.Dataset.from_pandas(df)


# +
repo_name = f"{username}/dummy-datasets"
dataset_name = f"dummy-dataset-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Create a dataset repository
hf_api = hf.HfApi()
try:
    hf_api.create_repo(repo_name)
except Exception as e:
    print(f"Error creating repository: {e}")

# Upload the dataset to the repository
dataset.push_to_hub(repo_name, config_name="full", private=True)
# -


# Now let's download the dataset we just created
dataset = datasets.load_dataset(repo_name)

dataset

df = dataset['train'].to_pandas()
df

df_every_other = df.iloc[::2]
df_every_other

dataset_every_other = datasets.Dataset.from_pandas(df_every_other)

dataset_every_other

dataset_every_other.push_to_hub(repo_name, config_name="every-other", private=True)

datasets.get_dataset_config_names(repo_name)

cyberpowder_org = "cyberpowder"
for ds in hf_api.list_datasets(author=cyberpowder_org):
    print(ds.id)

for ds in hf_api.list_datasets(author=username):
    print(ds.id)

for ds in hf_api.list_datasets(author=cyberpowder_org):
    print(datasets.get_dataset_config_names(ds.id))

# %%shell
# ls


