# tf-ludwig-google-cloud-ml-engine
This repo will train the model on google cloud ml engine using ludwig and host the model in ml engine.

### Installation
Ludwig has been developed and tested with Python 3 in mind. If you don’t have Python 3 installed, install it by running:

```
# on ubuntu
sudo apt install python3 

# on mac 
brew install python3 

# install ludwig
pip install ludwig
python -m spacy download en     
```

In this repo, I will try to build a simple text-classification using ludwig


## Using local python
You can run the code locally

```
JOB_DIR=jobDir
TRAIN_FILE=./data/train/*
EVAL_FILE=./data/eval/*
TRAIN_STEPS=2000

cd  tf-ludwig-google-cloud-ml-engine/

python3.6 -m trainer.task --train-files $TRAIN_FILE \
                       --eval-files $EVAL_FILE \
                       --job-dir $JOB_DIR \
                       --train-steps $TRAIN_STEPS.
'''