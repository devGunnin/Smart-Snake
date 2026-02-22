## Context
I want to write a scalable snake game. That means, a snake game that can have multiple users (lets say 4 users). The point of this snake game is every snake game. you eat apples and you grow. Now, the largest snake that survives wins. If you run into another snake tail you die. 

Now, this is supposed to have a good infrastructure. I.e. front-end that connects to some python webserver / API, that is then hosted via a backend. It should be implemented in a super smart way so that we can handle a lot of 4 player, 3 player, and 2 player multigames.

Now, in order to test the system, while doing something cool. I want to create a deep q network reinforcement learning that plays against it self. This should really be a state of the art reinforcement learning ML model that sits and plays itself in many multiplayer games to get better and better. I guess it should be a convolutional as its a game. It needs to have the correct state to understand the game and make wise decisions. 

Finally, we should be able to play against various versions of this model -- i.e. beginner, easy, medium, hard, impossible, etc.

This is the general project structure. It should inherit components of making a scalable game. Testing that via running many parallel games to train a DQN that eventually becomes an unbeatable snake player. The front-end that is developed at the end should be pretty. Finally, considerations of the grid size etc should be made in order to make it a fun and actually playable game. Computational cost is extremely important as it allows us to train faster. That means, well-written code, computationally efficient code, and good architecture and scalable architecture is of utmost importance.


## Problem
I want to write a scalable snake game. That means, a snake game that can have multiple users (lets say 4 users). The point of this snake game is every snake game. you eat apples and you grow. Now, the largest snake that survives wins. If you run into another snake tail you die. 

Now, this is supposed to have a good infrastructure. I.e. front-end that connects to some python webserver / API, that is then hosted via a backend. It should be implemented in a super smart way so that we can handle a lot of 4 player, 3 player, and 2 player multigames.

Now, in order to test the system, while doing something cool. I want to create a deep q network reinforcement learning that plays against it self. This should really be a state of the art reinforcement learning ML model that sits and plays itself in many multiplayer games to get better and better. I guess it should be a convolutional as its a game. It needs to have the correct state to understand the game and make wise decisions. 

Finally, we should be able to play against various versions of this model -- i.e. beginner, easy, medium, hard, impossible, etc.

This is the general project structure. It should inherit components of making a scalable game. Testing that via running many parallel games to train a DQN that eventually becomes an unbeatable snake player. The front-end that is developed at the end should be pretty. Finally, considerations of the grid size etc should be made in order to make it a fun and actually playable game. Computational cost is extremely important as it allows us to train faster. That means, well-written code, computationally efficient code, and good architecture and scalable architecture is of utmost importance.
