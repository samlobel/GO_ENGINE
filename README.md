# OmegaGo
The final word in Go AIs

___
### What is this?
It's a Go AI, written in Python with Tensorflow. Pretty different from AlphaGo. In some ways better, in most ways worse. FAQ follows, with a link to another writeup at the bottom. Comes with a GUI I built so that I could play against my bots.

___

### How do I play?
Download the repo, isntall the requirements (`pip install -r frozen.txt`), navigate to the folder GUI, and type `python go_gui.py`. It's that easy!

___

### How's it work?
It's sort of based off of AlphaGo. The original iteration had a single value function (a 4-layer CNN), which was trained through self-play. To choose a move, it simulates every possible next state, calculates the value of this next state, and chooses the move that results in the highest value.

The newer version is the same, except that instead of one value network, there are three. One is for the beginning of the game, one is for the middle, and one is for the end. It chooses which network to use based on the turn number.

___

### Why the three value functions?
It's a lot to ask of a network to output a sensible move during the beginning of the game compared to during the end of the game. CNNs work on extracting patterns from a board, and the patterns are WAY different on an empty board and on a full board. Having a smaller training scope means that a smaller network can better classify its inputs.

Plus, there are a few training advantages as well. Imagine having a perfect endgame value-function. Then, to find the value of a middlegame board, you only need to simulate until the endgame, not until the actual end. The same goes for the beginning-game versus middlegame.

In essence, training the networks from last to first means you're never more than 10 moves from a training signal, instead of 30 moves away. There's some way that this is an exponential improvement.

You couldn't do this on a single value function, because training on anything but boards in random order violates the IID hypothesis.

___

### Why no policy function? 
A few reasons. First is, policy functions are REALLY hard to train with no supervised data (like AlphaGo had). Value functions are easy to train, because there's intrinsic value to boards. For exapmle, there are some boards that are hard to lose from, no matter what you do. Even an untrained network can simulate a proper value for these. Choosing whether a specific move was good, on the other hand, involves trying out every move, and figuring out some ralative "goodness" for each of them. That's a WHOLE lot more simulation.

The second is, the policy function is strictly necessary when doing tree search (otherwise, at every node, you would need to test EVERY move's value). Luckily, I don't do tree search.

___

### But you need a policy function to simulate games off-policy, don't you?
It's helpful, but you can simulate a policy function with a value function. To do this, you look ahead to every legal move, assign it's move-space a value, and then push that value-board through a softmax function. These are your probabilities for your policy function. Using a higher softmax-temperature means more randomness (good for beginning stages of training), and a lower tempurature is more stringent (better for later stages).

### How do you train?
There were two sections to training: board generation, and regression.
##### Board generation
First, you play a game from start to finish, using the aforementioned policy-simulation, and record the winner. You then extract a single board from that game, along with the person who played on that board, and the board's turn number. Serialize, rinse and repeat, around 250,000 times. Lower the temperature as you train.

##### Regression
Starting with your list of boards and their values and metadata, it's as simple as creating the proper network inputs from this, and performing standard regression to make your network output similar values.

##### Putting it all together
I start out by thoroughly training the endgame value-function. Once that was good, I trained the middlegame, and then the beginning. I repeated training this way a few times, each time lowering the policy-temperature. The result? Some pretty darned good Go networks.

___
# I want more!
Look [here](./paper_outlines.md), at my writeup for the class that I started this project for. It's not perfect, but there's a good motivation and theory section, and some pretty formulas. Beware, it's a little outdated, because I've iterated a few times since then.
