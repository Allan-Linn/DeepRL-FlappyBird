# DeepRL-FlappyBird

Pytorch implementation of Deep Q-Networks (with different operators) for Flappy Bird environment. Deep Q-Learning has been shown to suffer from an overestimation problem due to the use of the max operator (due to Jensen's inequality and convexity of the max). Multiple replacements for the max operator are tested in the Flappy Bird environment. 


[Mellowmax] (https://arxiv.org/pdf/1612.05628.pdf)

[Adaptive Mellowmax] (https://cs.brown.edu/people/gdk/pubs/tuning_mellowmax_drlw.pdf)

[Soft Mellowmax] (https://arxiv.org/pdf/2012.09456.pdf)



