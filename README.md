![alt text](https://i.ibb.co/5jf3wPM/quarto-RL-logo.png)
# Quinto: Solving Quarto with Reinforcement Learning
This repository contains the code necessary to design an end-to-end experimental setting to train a Deep Reinforcement Learning (RL) agent playing the game of Quarto. 

The agent presented in this work is trained leveraging advancements made on On-Policy algorithms and, in particular, the training procedure does employ fairly innovative features such as Action Masking and incremental self-play.

# The game of Quarto
Quarto is a two-player board game that involves strategy, logic, and luck. The goal of the game is to be the first player to create a line of four pieces, either horizontally, vertically, or diagonally. The game pieces consist of 16 unique pieces with four distinct attributes: shape, height, color, and filling. 
More formally:

<p align="center">
    <img width=400 src="https://i.ibb.co/T05Hc2P/quartopieces.jpg" alt="pieces">
</p>
<p align="center">
    <em>Quarto Pieces sharing a common property</em>
</p>


1. The game of Quarto is played by two players using a 16 piece game set, which consists of 4 by 4 square pieces. Each piece has four characteristics: **height**, **width**, **shape**, and **color**.

2. The purpose of the game is for one player to form a row of four pieces, each with at least one shared characteristic, while the other player tries to prevent it.


3. Players alternate turns, with the starting player determined by a coin toss. During each turn, the player must select one of the ones not on the board and offer it to their opponent, who then must place it in a cell of their (the opponent's) choice. The goal of each player is to form a line of four pieces sharing one property, so the piece position is typically chosen according to this aim.

4. If the game reaches a stalemate, with all possible moves blocked, the game is declared a draw.

More information about the game and how to play can be found [at this page](https://en.wikipedia.org/wiki/Quarto_(board_game)) or at [this YouTube video](https://www.youtube.com/watch?v=v1c-uKD6iOw&themeRefresh=1).

# Installation
The project requires Python 3.7 or higher and, among many, the following packages:

- `stable-baselines3`
- `sb3-contrib`
- `gym`
- `torch`

To reproduce our training routine you must first create a virtual environment in which to install the required dependencies. You can do this via the following: 
```
$ conda create -n quarto
$ pip install -r requirements.txt
```

# Usage

## Reproduce our results
To train the agent with default configuration, simply run the following command:

```python
$ python train.py
```

The `train.py` scripts makes usage of several arguments. For the sake of brevity, we do not report each of them here. However, you can read more about them by simply running `python train.py -h`.

### Re-train our best performing agent

**Note:** the default configuration (the one used to train our best-performing agent) produces **`MASKEDPPOv2_100e6.zip`**. This model must later on be fine-tuned with self-play to finally obtain **`MASKEDPPOv3_120e6.zip`**, our best performing agent.

This procedure can be carried out running the following: 

```bash
$ python train.py  # train MASKEDv2_100e6, takes approximately a week.

$ python train.py --default False --algorithm maskedPPO --train-timesteps 20e6 --test-episodes 100 --losing-penalty True --save-model True --duration-penalty True --action-masking True --model-path commons/trainedmodels/MASKEDv2_100e6.zip --use-symmetries True --self-play True --resume-training True  # insert 20e6 training timesteps when prompted
```

Should the training process of `MASKEDPPOv2_100e6.zip` crash (as it has been crashing on our machine) you can resume training from one of the checkpoints saved in the `checkpoints` folder, by simply running: 

```
$ python train.py --default False --algorithm maskedPPO --test-episodes 100 --losing-penalty True --save-model True --duration-penalty True --action-masking True --model-path checkpoints/<checkpointed_model> --resume-training True # insert the number of timesteps you want the model to train after crashing
```

In a nutshell, the arguments of `train.py` are used to differentiate the training process for what concerns the **reward function** used and the **opponent** challenged during the training process. 

Furthermore, a version of the training process which makes usage of the highly symmetric structure of the game is also available. A precise description of every version implemented so far is presented in the following table. 

| **_Environment version_** |            **_Reward Function_**            | **_Uses symmetries_** |       **_Opponent_**       |
|:-------------------------:|:-------------------------------------------:|:---------------------:|:--------------------------:|
|            `v0`           |             $r_T = (+1, +0.2, 0)$            |           No          | Valid-Moves-Only Random Opponent |
|            `v1`           |            $r_T = (+1, +0.2, -1)$            |           No          | Valid-Moves-Only Random Opponent |
|            `v2`           | $r_T = (5 - 0.75 \cdot (T/2 - 2), +0.5, -1)$ |           No          | Valid-Moves-Only Random Opponent |
|            `v3`           | $r_T = (5 - 0.75 \cdot (T/2 - 2), +0.5, -1)$ |  Yes, rotational only |          Self-play         |

Where $r_T$ indicates the reward obtained by the agent in the terminal state (namely, a state in which it wins, loses or draws the game). The elements of $r_T$ represent the reward obtained for winning, drawing and losing (respectively) a training game. 

It is worth mentioning that $r_t = 0 \ \forall t \neq T$ ( $r$ is, thefore, a *sparse* reward). 

Since we wanted our agent to be as general as possible, we avoided providing intermediate rewards for reaching certain board configurations. 
This was a clear design choice that, while clearly made the training process way more complex, prevented us to bias the agent towards a particular (not necessarily optimal) behavior.

During the training process, the agent would be periodically saved in a `checkpoint` folder (that would also be used to load periodically dump and load self-playing agents). 

Once a model is fully trained, it is saved in the `commons/trainedmodels` folder. During the training process the policy trained is optionally tested againts its own opponent for an arbitrary number of games, in the sake of more accurately diagnosing the characteristics of the training process. These data files are stored in the `logs` file.

Models are saved in the `<ALGORITHM><VERSION>_<TRAININGTIMESTEPS>.zip` format, whereas logs are saved in the `<ALGORITHM><VERSION>_<TRAININGTIMESTEPS>_logfile.txt`.

For the sake of completeness, we also report here the time needed to train these models.


| **_Algorithm_** | **_Version_** | **_Timesteps_** | **_Training time_** |
|:---------------:|:-------------:|:---------------:|:-------------------:|
|     **PPO**     |       v0      |       5e6       |       _6h 29m_      |
|     **PPO**     |       v1      |       5e6       |       _4h 34m_      |
|     **A2C**     |       v0      |       5e6       |       _5h 43m_      |
|     **A2C**     |       v1      |       5e6       |       _4h 52m_      |
|  **MaskedPPO**  |       v0      |       5e6       |       _6h 27m_      |
|  **MaskedPPO**  |       v1      |       5e6       |       _7h 28m_      |
|  **MaskedPPO**  |       v2      |      100e6      |       _~1 week_      |
|  **MaskedPPO**  |       v3      |   (100 + 20)e6  |   _~1 day, 3 hours_  |


Please note that the last model presented here is nothing but an instance of the the `MASKEDPPOv2_100e6` model incrementally trained with self-play and symmetries for an additional 20M timesteps.

More details on the training procedures and the rationale behind the algorithm choices can be found in our report.

Our best performing model can be found at: `commons/trainedmodels/MASKEDPPOv3_120.zip`. 

## Loading a trained model

Depending on the model used, it is necessary to use different objects (either from `stable-baselines3` or `sb3-contrib`) to correctly load the state dict characteristic of each model.

To load a A2C or PPO-based model simply run (as per the official documentation): 

```python
from stable_baselines3 import A2C, PPO
from sb3_contrib import MaskablePPO
from commons.quarto.objects import Quarto
from main import RLPlayer


A2C_model = A2C.load("commons/trainedmodels/A2Cv1_5e6.zip")
PPO_model = PPO.load("commons/trainedmodels/PPOv0_5e6.zip")
maskedPPO_model = MaskablePPO.load(
            'commons/trainedmodels/MASKEDPPOv3_130e6.zip',
            custom_objects= {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0, 
                "clip_range": lambda _: 0.0
            }
)
# create instance of the game 
game = Quarto()
# create player
playerRL = RLPlayer(game, maskedPPO_model)  # can also be A2C_model or PPO_model
```

The fact that models are wrapped by the `Player` class (defined in `commons/quarto/objects`) is justified by the fact that we wanted our players to be compatible with the common interface presented [here](https://github.com/squillero/computational-intelligence/tree/master/2022-23/quarto) and reproduced in this repo in the `main.py` file.

Our players are modelled as instances of the `RLPlayer(...)`.

## Experiments & Results
An extensive discussion of our experimental results can be found in the full report that complements this repo. 

Our experiments related to the training phase of `v3` can be found [here](https://wandb.ai/francescocapuano/QuartoRL-v3%20training?workspace=user-). 

Currently, we are still experimenting with a last version fully trained with self-play for a larger number of episodes. 
These experiments can be consulted [here](https://wandb.ai/francescocapuano/QuartoRL-v2%20seedless%20training?workspace=user-). 

Tested against a randomly playing agent, our algorithm peaks 90%+ winning rate over 100 games. More interestingly, however, our approach allows running **more than 50 games a second**.

In particular, our algorithm can play 1000 games against a random agent, win the vast majority of these matches while requiring 50 seconds only. 



## Credits
While we take pride in the full pathernity of this work, we would like to acknowledge the influence of several online open-source resources such as: 

1. [This repo](https://github.com/benallard/quarto-gym) has been used as an initial skeleton to develop this project. While we significantly expanded its code-base, having it allowed us to start on the right foot.

2. [This piece of documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html) helped us developing the action-masking framework we have been using in this report.

3. [This repo](https://github.com/hardmaru/slimevolleygym/blob/master/TRAINING.md) played a major role in the implementation of our self-play paradigm.

4. As usual, countless visits to [Stack Overflow](https://stackoverflow.com/) have been a key part of the development process.

5. [This paper](https://www.researchgate.net/publication/261848662_An_artificial_intelligence_for_the_board_game_'Quarto'_in_Java) has helped us in better framing the problem. 

6. Lastly, [this book](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) also played a huge role in our RL-based approach to this problem.

7. Lately, we have been experimenting with [ChatGPT](https://chat.openai.com/) for brainstorming and implementation tweaks.

# Authors
Francesco Capuano ([@fracapuano](https://github.com/fracapuano)), Matteo Matteotti ([@mttmtt31](https://github.com/mttmtt31))
