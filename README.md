# Dogs vs. Cats Kaggle's competition project

The purpose of this project is to develop a solution for the Kaggle's competition dogs-vs-cats (https://www.kaggle.com/c/dogs-vs-cats). As a learning exercise, the solution will be more and more complex developing things like a MLaaS API, a package at PyPi, etc. Another purpose is to maintain a branch with only (almost) Pytorch and others with higher level of abstraction, like using Pytorch-Lightning. The main branch contains the latest development.

## Quick start
If you want to train or eval the models you need to download first the competition data from Kaggle. Also, check the config script for some key config aspects and paths.
### Training
For train your models you can use the train script:
```
python dogsvscats/train.py -h
usage: train.py [-h] [-m {resnet18,mobilenet_v3_small}] [-cp CHECKPOINT_PATH] [-w WORKERS] [-bs BATCH_SIZE] [-lr LEARNING_RATE]
                [-e EPOCHS] [-sp SCHEDULER_PATIENCE] [-esp EARLY_STOPPING_PATIENCE] [-d] [-df DEBUG_FRAC] [-vf VALID_FRAC]

optional arguments:
  -h, --help            show this help message and exit
  -m {resnet18,mobilenet_v3_small}, --model {resnet18,mobilenet_v3_small}
                        Model name
  -cp CHECKPOINT_PATH, --checkpoint-path CHECKPOINT_PATH
                        Checkpoint Path
  -w WORKERS, --workers WORKERS
                        Workers
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate
  -e EPOCHS, --epochs EPOCHS
                        Epochs
  -sp SCHEDULER_PATIENCE, --scheduler-patience SCHEDULER_PATIENCE
                        Scheduler patience
  -esp EARLY_STOPPING_PATIENCE, --early-stopping-patience EARLY_STOPPING_PATIENCE
                        Early stopping patience
  -d, --debug           Debug
  -df DEBUG_FRAC, --debug-frac DEBUG_FRAC
                        Debug fraction
  -vf VALID_FRAC, --valid-frac VALID_FRAC
                        Validation fraction
```

### Evaluation
For evaluate your models you can use the eval script:
```
python dogsvscats/eval.py -h 
usage: eval.py [-h] [-m {resnet18,mobilenet_v3_small}] [-cp CHECKPOINT_PATH] [-w WORKERS] [-bs BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -m {resnet18,mobilenet_v3_small}, --model {resnet18,mobilenet_v3_small}
                        Model name
  -cp CHECKPOINT_PATH, --checkpoint-path CHECKPOINT_PATH
                        Checkpoint Path
  -w WORKERS, --workers WORKERS
                        Workers
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
```

### Prediction
For predict an image, using already trained models (-d) or yours (-cp) you can use the predict script:
```
python dogsvscats/predict.py -h
usage: predict.py [-h] [-i IMAGE] [-m {resnet18,mobilenet_v3_small}] [-cp CHECKPOINT_PATH] [-d]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        Input image
  -m {resnet18,mobilenet_v3_small}, --model {resnet18,mobilenet_v3_small}
                        Model name
  -cp CHECKPOINT_PATH, --checkpoint-path CHECKPOINT_PATH
                        Checkpoint Path
  -d, --download        Checkpoint Path
```

![dogs-vs-cats-img](https://storage.googleapis.com/kaggle-competitions/kaggle/3362/media/woof_meow.jpg "Dogs-vs-Cats")