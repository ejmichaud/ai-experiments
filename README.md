```
                                                            _---~~(~~-_.
   ____     __      _____                                 _{        )   )
  /  _/__  / /____ / / (_)__ ____ ___  _______          ,   ) -~~- ( ,-' )_
 _/ // _ \/ __/ -_) / / / _ `/ -_) _ \/ __/ -_)        (  `-,_..`., )-- '_,)
/___/_//_/\__/\__/_/_/_/\_, /\__/_//_/\__/\__/        ( ` _)  (  -~( -_ `,  }
   ____                /___/               __         (_-  _  ~_-~~~~`,  ,' )
  / __/_ __ ___  ___ ____(_)_ _  ___ ___  / /____       `~ ->(    __;-,((()))
 / _/ \ \ // _ \/ -_) __/ /  ' \/ -_) _ \/ __(_->             ~~~~ {_ -_(())
/___//_\_\/ .__/\__/_/ /_/_/_/_/\__/_//_/\__/___/                    `\  }
         /_/                                                           { }

```

# What's here?
Various AI and Machine Learning demos. The goal is to have a well-organized, central location for exploring ideas in machine intelligence.

## Table of Contents
- [Universal Approximators](universal-approximators/universal-approximators.ipynb): Explores how neural networks can approximate arbitrary functions
- [Dense MNIST](mnist-dense/mnist-dense.ipynb): A quick example of classifying handwritten digits from the MNIST dataset.
- [Convolutional MNIST](mnist-cnn/mnist-cnn.ipynb): Using a CNN to classify MNIST digits.
- [Dimensionality Reduction](dimensionality-reduction/dimensionality-reduction.ipynb): Testing PCA and UMAP on MNIST and using clustering on the embeddings for classification.

## Getting things running

The most versatile way to start is by building a dedicated docker image. From the repository root directory, run:
```
docker build -t experiments .
```
And to run, use:
```
docker run --rm -it \
	--volume ./:/root/ai-experiments  \
	--publish 8888:8888 \
	source lab.sh
```
If you want to use a tensorflow notebook, you'll have to run run the container and install from inside. That way, you can specify whether to install `tensorflow-gpu` or plain `tensorflow`:
```
docker run --rm -it \
	--volume ./:/root/ai-experiments  \
	--publish 8888:8888 \
	/bin/bash

$ conda install -y tensorflow-gpu
$ source lab.sh
```

**`lab.sh`** executes:
```
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```

**To Eric**: To activate the python virtual environment on your good'ol trusty laptop, simply use the `mlenv` command, an alias declared in `~/.bash_profile`. 

If you are reinstalling this repo on another machine, create the venv like so:
```
virtualenv -p $(which python3) venv
source venv/bin/activate
pip install -r requirements.txt
```
Then create a new jupyter notebook kernel using the virtual environment like so:
```
python -m ipykernel install --user --name mlenv --display-name "Python 3 (mlenv)"
```
And launch the notebook with:
```
python -m jupyter lab
```
Make sure that you are run these last commands with the virtual environment activated.

## Experiment Ideas:
- [x] Get a simple neutal network-based regression to work.
- [ ] Investigate eliminating magnitude of gradient. It would just tell you which direction to increment. To get convergence, may need to decrease step size over time. 
- [ ] Explore ANN topology space by not resetting all parameters of the network - add just a few neurons at a time — slight changes to topology might allow 
for a kind of pseudo-transfer learning to happen. If parametrized elegantly, the space could be searched by a reinforcement learning or genetic algorithm. 
- [ ] ^^^ What if a network complexity score could be assigned to each point in the parameter space? Then the optimization/search algorithm could not just take into account accuracy, but also complexity, and attempt to create the most efficient (accuracy/complexity, or some more sophisticated cost function) network. 
