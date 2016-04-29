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




