## NNET GO


This is going to be a really, really involved project.


There's many things I need to do.

## TO INSTALL:
You need to make a virtualenv. If you have virtualenv installed, you can
make one with the command `virtualenv venv`, then activate it with `source venv/bin/activate`, and then install the requirements with `pip install -r frozen.txt`. Then, run the GUI by going to the GUI folder and running `python go_gui.py`.

There are some other things that won't work right now, because my TENSORFLOW model-values aren't on github (too big). 





### SEPARATE TASKS
#### TASK ONE
I need to make a GUI that you can play against, so you can see what's going on. That's going to be a process in itself.

#### TASK TWO
I need to make a CNN that takes in a GO board (formatted as B=1,W=-1,EMPTY=0), and outputs a value. I should make something like 10 of them, one for each stage of the game. They should match up on the edges if possible. I should also probably use ReLu units, or the ones that are negative-exponential for negative and linear for positive.

Should I use a CNN? On one hand, there are very strong relations between neighboring pieces. On the othe hand, it's a pretty small board (smaller than MNIST by a factor of two!) and I have technically infinite training data.

I think I won't, I'll just use highway neurons and lots of layers. BUT, I feel like 361 features just isn't enough for a game of go. It's gotta expand at first. On the other hand, highwaying 361 might be fine. 

##### Best yet, if I make class interfaces, I'll just be able to sub in one network for another. This is the answer, the best answer yet. Oh boy, I like good answers. Time to become a good programmer Sam.

Sort of a simple alternative to CNN is augment the feature vector, for each point adding something like the value of the number of free spaces it has around it, or eight features per space that are its relation to the eight surrounding pieces. That feels hacky, and I'm not sure exactly how it would work. I guess CNNs it is.

A problem I have with all of this is, there's counting in GO, and there's no real good way for neural networks to count. Maybe that means I need to use a LTSM, because they definitely can count. That's a great idea, Sam.


#### TASK THREE
I need to write a value-update function. Given a board, it tries every possible move (including not moving), and then does a continuous search from there. It travels all the way to the end, playing itself, and then updates the value to be a little more like the
true end value of the best choice.


#### TASK FOUR
I need to write a best-move function, that tests the value of every move and picks the best one. That's easy, once we've got the NN

#### TASK FIVE
I should write an interface for this to the artificial go servers

#### TASK SIX
I should test it on the go database, that checks for best-moves.


### Thoughts on ways to do this well.
It seems like a good way to do training is to generate board positions that are ties, because that's not a real option. We'd want to push these boards towards winning or losing. There's two things about this: First, I need to look at CNN generation more closely. Second, I need to figure out a way to put a sparsity constraint on this, because I want to be able to sample the state space of time-in-game well.

Also, it seems like making different CNNs for different board times is going to be a pretty tough ordeal. So, I'll start out by just having one, but hopefully I code it in a way that allows me to use subclasses of a model board.

Also, since we're doing +1/-1 for black/white, I may need to think closely about how I want my layers to look. Because negative values aren't helpful with ReLus. On the other hand, maybe the weights, and having multiple channels, will solve that. Mainly the multiple channels part.



I should also have a few different guys that I train. Maybe a very shallow NNet to start.

I really want to figure out how to generate boards of a certain value. Look up something called deconvolutional neural networks

Chinese scoring.

NOT DONE: I'M WRITING A SPOT-WON-BY FUNCTION, THAT DETERMINES WHETHER A SPOT IS CONTROLLED
BY ONE SIDE OR THE OTHER.

Occasionally, I should do something that enforces symmetry, so that it's on the right path. I could easily generate symmetric games, and force both scores towards the average.
That would be great actually, because there are 8 symmetries, which is a great way to make sure it really learns the meaning of go.



AN INTERESTING IDEA: is to have two channels going in, which are the color of the spot, and
the liberties of that spot. That's not a bad idea. Especially because liberties are so strictly positive. That's like the perfect use of CNNs and channels. I think I'll experiment with that. Should I do absolute value of eyes? Or should I do liberties times
color? I think that's a better idea.


WHERE I LEFT OFF:
LOOKS LIKE THINGS ARE GOING WELL, EXCEPT THAT THE INFINITE LOOP CONDITION IS NOT BEING CAUGHT QUITE PERFECTLY. IM SURE I CAN FIGURE OUT WHY LATER.


I should make my program adhere to the GTP (Go Text Protocol).
http://cgos.boardspace.net/

Therre may be some stuff about black wanting to maximize value and white wanting to minimize it.


Looks like I use tail recursion way too much, and its breaking my program. That stinks.
I should also try and find ways to make this faster, its slow as fuck.


I'm scared I have a bug. It's learning, but it's not really getting any better, and that's not good. It seems like I could have really easily messed up in my thought process for the flipping the board, but I don't see exactly how. Maybe I should do the board-flip before the apply_move thing.


I think maybe I've been doing this all wrong. Maybe what I need to do is
temporal back-up, I could be doubling down on a bad policy. But, I do think that what I'm doing should converge.

What I'm doing:
I see a board, and I play it to its end. I also get the board's 'value'. If the
play-to-end wins, then I increase this board's 'value', saying "I want to be in states like this one." If the play-to-end loses, then I decrease the value, saying "I don't want to be in states like this, because my policy loses at them."

I want to see my classification accuracy. Maybe the problem is that
my thing has converged.



I need to simulate a board, then try EVERY possibility from there, and then update the value functions for each of the possibilities. That way, if there's a better choice somewhere than I thought, the algorithm figures that out and will be more likely to pick it next time.


I could use the value function to update every single spot it sees, that would
be a huge speedup.


Instead of random games, and then on-policy, I should go on-policy for n steps,
and then simulate every move, and continue those from there on-policy. Then update the value for every state that follows the one you broke policy on.


In determine winner, have handicap option.

To determine score, I don't just need to count the ones surrounded by only white, it should be the ones surrounded by a SINGLE group.


http://www.cosumi.net/en/   is a great resource.

By the way, I really should make the network deeper, as well as pass in the total liberties per group as an input. It's a great idea.

I need to make a policy network for training. When it picks the next move, 
it should go based off of percentages given by the policy network.
And the policy network should learn from the value network.

How should the policy network output its answers? Softmax temperature!
That's super interesting, it's perfect. But, it looks like there's no option
for it, so I'm going to need to code my own softmax function.

And it can go both ways! The softmax should be used in picking moves for simulation, but also when you train the policy network based on the value
network, you can try and shoot it so that the policy network learns probabilities in proportion to softmax of the values.

The more annoying thing is, I'm not exactly sure how to output the answer from the softmax.

I should train the policy network using the values of the moves. I guess that the policy network should just output a string that's 26 long or something (5x5 + passing). That's sort of a weird thing to do, but whatever. It would be much nicer if we could use something convolutional.


I'm ending at not-the-best spot. I'm sort of in the middle of the process of adding features, it's been messsy withthe dimension changing part of it. BUT,
it's going to start working again


I really can't get past this problem with the AdamOptimizer. It seems like a big design problem. On the other hand, it's WAY better for convergance, especially when I'm going to have policy gradient stuff. Honestly, I'm lost.

##### SOLVED!
The AdamOptimizer is tough because you can't namespace the internal variables. But, you don't need to use the optimizer for playing, only training. And the 
training only happens against itself anyways. So, I have a TEST_OR_TRAIN
flag. When I want to use the GUI, I set it to TEST, and it doesn't load the
optimizer, and everything else is namespaced. All good baby.



### SOMETHING INTERESTING
When I use the value network, I'm always asking the board where white is about to go. The reason this is true is, I only care about black, and I use the value network after I have tried every valid move. So, black goes and I evaluate how likely black is to win, given that white is about to move.
On the other hand, the policy network also helps black make moves. So the policy network always assumes that it's about to be BLACK's turn, and then you do a move that is most likely to help BLACK. So, they sort of operate on different boards.

After you play a game out, you get true end values. You then use those values as energies, then pass them through a softmax, and then update the POLICY network based on this softmax. 

Two things I could play with: first, I could use either the estimated board-values before the update or after. Second, I could use a slightly lower temperature for the update to the POLICY network.


### Why what I'm going to do will work.
The value function updates based on the policy. It says, "if I play decently
well from here, how likely is it that I'll win?" Then, it gets better and better valuations of boards. The policy function is a way of playing well. You make it play better by matching the value function as well as possible.
As we train, we make the policy tighter and tighter against the value. Nice.

I should make an intitial training function that learns what is a legal move and what isnt. This is sort of like learnign the policy with a very high temperature. This shouldn't be too hard, because the training data is very quick to produce, so I could do it in large batches.


scp -ri /Users/samlobel/.ssh/PEM/TensorflowEC2.pem ubuntu@ec2-54-149-232-179.us-west-2.compute.amazonaws.com:/home/ubuntu/GO_ENGINE/NNET/FIVE/saved_models ~/saved_models
thats how I get the models over.


It's making it way more difficult because of the whole 26 output thing. It would make it a lot easier if there was a simple rule about 'are there sensible moves'. It should be easy to code that, if a move fills an eye it's bad, and if a move is illegal it's bad as well. And that way, you never need to output None!

CLEARLY THIS ISN'T WORKING.
My model should most definitely be improving after so much time training. But it's not.

I could: just do value iteration. That's a possibility. The other possibility is that I could just do policy learning. Value learning is a little sexier to me, because theres not this problem of outputting None.


If I have a proper understanding of this, what it's saying is: 

You have a policy. It has parameters. When you lose a game, you don't know exactly where you went wrong. But you do know that you made a mistake somewhere along the line. So, you made a bunch of decisions during the game, and you want to update it so that these specific decisions are a little bit less likely to be made. Or more likely, if it was a win. 


So, maybe I could almost-all-zero-multiply the thing so that only the one I care about is showing, and then minimize that guy squared.

That's an interesting approach. What I do is, I want to make the probability of picking that one action smaller, so what I do is, if I lost with a specific action, 
minimize(sum(output*zero_masks_for_all_but_the_guy_we_like))

Or, if it was right, we could minimize the negative of that.

That's super wasteful though, I feel like there must be a better way. I think dealing with the raw output would be a good place to start, because softmax does some funky stuff, but you can edit the underlying thing without worrying about reapportioning.

Anywho, that's not a bad idea.

I just want to decrease the chance of doing something if its bad, or increase it if its good. So, I compute the 


I love that you can do interpreter-level tensorflow stuff. It rocks, its so much easier than theano.
So, here's what you do.
You play a game from start to finish, on-policy (with some temperature).
If you win,



## TO DO TOMORROW 
I need to write something that will continuously simulate, and do rounds.
It should be 100 games per round, against a random opponent. I can have a file
in each folder that tells you the latest bout that's been successful.
And that's that!  



