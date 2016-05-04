## NNET GO


This is going to be a really, really involved project.


There's many things I need to do.

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



