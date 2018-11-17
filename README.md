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
Various AI and Machine Learning experiments. The goal is to have a well-organized, central location for exploring ideas, original and otherwise, around these topics. 

## Getting things running
To activate the python virtual environment on the good'ol trusty laptop, simply use the `mlenv` command, an alias declared in `~/.bash_profile`. If you are reinstalling this repo on another machine, create the venv like so:
```
virtualenv -p $(which python3) venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```
## Experiment Ideas:
- [x] Get a simple neutal network-based regression to work.
- [ ] Investigate eliminating magnitude of gradient. It would just tell you which direction to increment. To get convergence, may need to decrease step size over time. 
- [ ] Explore ANN topology space by not resetting all parameters of the network - add just a few neurons at a time — slight changes to topology might allow 
for a kind of pseudo-transfer learning to happen. If parametrized elegantly, the space could be searched by a reinforcement learning or genetic algorithm. 
- [ ] ^^^ What if a network complexity score could be assigned to each point in the parameter space? Then the optimization/search algorithm could not just take into account accuracy, but also complexity, and attempt to create the most efficient (accuracy/complexity, or some more sophisticated cost function) network. 
