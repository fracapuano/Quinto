![alt text](https://i.ibb.co/5jf3wPM/quarto-RL-logo.png)
# QuartoRL: Solving Quarto with Reinforcement Learning
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

To reproduce our training routine you must first create a virtual environment in which to install the required dependancies. You can do this via the following: 
```
conda create -n quarto
pip install -r requirements.txt
```

# Usage

## Reproduce our results
To train the agent, simply run the following command:

```python
python train.py
```

The `train.py` scripts makes usage of several arguments. For the sake of brevity, we do not report each of them here. However, you can read more about them by simply running `python train.py -h`.

In a nutshell, the arguments are used to differentiate the training process for what concerns the **reward function** used and the **opponent** challenged during the training process. Furthermore, a version of the training process which makes usage of the highly symmetric structure of the game is also available. In particular: 

| **_Environment version_** |            **_Reward Function_**            | **_Uses symmetries_** |       **_Opponent_**       |
|:-------------------------:|:-------------------------------------------:|:---------------------:|:--------------------------:|
|            `v0`           |             $r_T = (+1, +0.2, 0)$            |           No          | Valid-Moves-Only Random Opponent |
|            `v1`           |            $r_T = (+1, +0.2, -1)$            |           No          | Valid-Moves-Only Random Opponent |
|            `v2`           | $r_T = (5 - 0.75 \cdot (T/2 - 2), +0.5, -1)$ |           No          | Valid-Moves-Only Random Opponent |
|            `v3`           | $r_T = (5 - 0.75 \cdot (T/2 - 2), +0.5, -1)$ |  Yes, rotational only |          Self-play         |

Where $r_T$ indicates the reward obtained by the agent in the terminal state (namely, a state in which it wins, loses or draws the game). The elements of $r_T$ represent the reward obtained for winning, drawing and losing (respectively) a training game. 

It is worth mentioning that $r_t = 0 \ \forall t \neq T$ ($r$ is, thefore, a *sparse* reward). 

Since we wanted our agent to be as general as possible, we avoided providing intermediate rewards for reaching certain board configurations. 
This was a clear design choice, that while clearly made the training process way more complex, prevented us to bias the agent towards a particular (not necessarily optimal) behavior.

During the training process, the agent would be periodically saved in a `checkpoint` folder (that would also be used to load periodically dump and load self-playing agents). 

Once a model is fully trained, it is saved in the `commons/trainedmodels` folder. During the training process the policy trained is optionally tested againts its own opponent for an arbitrary number of games, in the sake of more accurately diagnosing the characteristics of the training process. These data files are stored in the `logs` file.

Models' are saved in the `<ALGORITHM><VERSION>_<TRAININGTIMESTEPS>.zip` format, whereas logs are saved in the `<ALGORITHM><VERSION>_<TRAININGTIMESTEPS>_logfile.txt`.

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

More details on the training procedures and the rationale behind the algorithm choices can be found in our report.

## Loading a trained model

Depending on the model used, it is necessary to use different objects to train the state dict that characterizes 

## Results
The agent was able to achieve an average win rate of ~65% after 10,000 games.

## Credits
This project was implemented by [Your Name].