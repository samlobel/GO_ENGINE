Alright, here goes nothing.
### Moving forward, gameplan.

Thing number one is clean things up a little.

Thing number two is change it so that it only looks back a single move for Koh. This will speed it up a lot.

Thing number three is, work with handicaps, which means passing in a "this persons color" layer. 

Thing number four is, design it so there are three bots or something, and break it up based on turn number (or number of things on the field. But I like turn number better).

Then, design evaluation function that will play using all of the boards

Then, write better coverage of board generation. That's important for training.
Especially for training backwards.


I think I should only have a value function. Policy doesn't make sense.
Especially because you can train policy easily at the end.

I should make it SUPER modular in terms of board size. 

Alright, here it goes....