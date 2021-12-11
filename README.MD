								
# Comic Generator using GPT2 

Directory Structure: 

Application: Contains Story Gneration notebook deployed on Vertex AI using custom pipeline.

Notebooks: Story generation colab and colab with image generation on top of story generation.

# Fine Tuning GPT2

GPT-2 is a language model comes from OpenAI’s paper “Language Models are Unsupervised Multitask Learners”. GPT was a variation of the famous Transformer architecture proposed by the Google Brain team in their paper “Attention is all You Need” . Fine tuning it can enable us to write long and engaging stories. 

We using this projet tried to achieve just that. WritingPrompts is a dataset published by Facebook AI Research intending to make machines able to compose a long and consistent story we used that to generate long form text or stories usoing GPT2.

Perplexity score before fien tuning the model: 

![image](https://user-images.githubusercontent.com/10840984/145663578-919f59cc-5edd-4c9b-9ee6-c7de87ba6f51.png)

The perfromance of the fine tuned network 
![W B Chart 12_10_2021, 5_55_38 PM](https://user-images.githubusercontent.com/10840984/145663418-40904454-3eef-40f0-88fb-5c83ae2d353e.png)

# Image Generation using Deep Daze library

Simple command line tool for text to image generation using OpenAI's CLIP and Siren. Helped us achieve good results for image generation from paragraphs of text.

![image](https://user-images.githubusercontent.com/10840984/145663899-3a9a3d26-9598-46ac-8c25-020473c4d271.png)

# MLOps using Vertex AI
In google cloud platform create a project and enable vertexAI API, Google cloud storage API, container registry API and notebook API. 

In google cloud platform create a pytorch notebook using the command below.

	gcloud notebooks instances create story-generation     --vm-image-project=deeplearning-platform-release     --vm-image-family=pytorch-1-9-cu110-notebooks     --machine-type=n1-standard-4     --location=us-central1-a     --boot-disk-size=100 --network=default
	
Create a bucket in your project and upload the dataset.

create a package with structure below run below command to create tar file and  upload to the bucket in google storage.

	python setup.py sdist --formats=gztar 
	-python_package
		-trainer
			-__init__.py
			-experiment.py
			-metadata.py
			-model.py
			-task.py
			-utils.py
		-steup.py
	-predictor
		-custom_text_handler.py
		-Dockerfile

Initialise the aiplatform with project id and location.

Initialise and run the job with CustomPythonPackageTrainingJob method with providing the container image and location to the package in the bucket. Training object saves the model artifacts in the output bucket in storage bucket.

Load the model artifacts to the local director in notebook and to create the custom predictor image using the Dockerfile. After building the package push it to the container registry.

Upload the trained model artifacts using the custom predictor image.

Create endpoint on vertex AI platform.

Deploy the model to the created enpoint and predict using the endpoint. 

FLASK APPLICATION:

Requirements :
pip3 install torch

git clone https://github.com/huggingface/transformers

pip3 install transformers/

pip3 install flask

run python3 deploy.py

application is served at localhost:5000/






