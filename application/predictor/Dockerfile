
FROM pytorch/torchserve:latest

# install dependencies
RUN pip3 install transformers
# RUN git clone https://github.com/pytorch/serve.git
# RUN pip3 install torchserve torch-model-archiver torch-workflow-archiver
RUN pip3 install torch torchvision torchserve torch-model-archiver -f https://download.pytorch.org/whl/torch_stable.html

# copy model artifacts, custom handler and other dependencies
# RUN pwd
# RUN ls -lart /home
COPY ./custom_text_handler.py /home/model-server/
COPY ./story-gen/ /home/model-server/

# create torchserve configuration file
USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
RUN printf "\ninference_address=http://0.0.0.0:9000" >> /home/model-server/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:9001" >> /home/model-server/config.properties
RUN printf "\nmetrics_address=http://0.0.0.0:9002" >> /home/model-server/config.properties
# RUN printf "\ninstall_py_dep_per_model=true" >> /home/model-server/config.properties

USER model-server

# expose health and prediction listener ports from the image
EXPOSE 9000
EXPOSE 9001

# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f   --model-name=story-gen   --version=1.0   --serialized-file=/home/model-server/pytorch_model.bin   --handler=/home/model-server/custom_text_handler.py   --extra-files "/home/model-server/config.json,/home/model-server/training_args.bin"   --export-path=/home/model-server/model-store

# run Torchserve HTTP serve to respond to prediction requests
CMD ["torchserve",      "--start",      "--ncs",      "--ts-config=/home/model-server/config.properties",      "--models",      "story-gen=story-gen.mar",      "--model-store",      "/home/model-server/model-store"]

