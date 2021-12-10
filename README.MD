								COMIC GENERATOR USING GPT2

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






