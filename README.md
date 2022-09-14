# DeepRL-FlappyBird

Pytorch implementation of Deep Q-Networks (with different operators) for Flappy Bird environment. Deep Q-Learning has been shown to suffer from an overestimation problem due to the use of the max operator (due to Jensen's inequality and convexity of the max). Multiple replacements for the max operator are tested in the Flappy Bird environment. 


Links to the original papers are embedded below:

[Mellowmax](https://arxiv.org/pdf/1612.05628.pdf):
Asadi, K., &amp; Littman, M. L. (2017, June 14). An alternative softmax operator for reinforcement learning. arXiv.org. Retrieved September 13, 2022, from https://arxiv.org/abs/1612.05628 

[Adaptive Mellowmax](https://cs.brown.edu/people/gdk/pubs/tuning_mellowmax_drlw.pdf):
“Adaptive temperature tuning for mellowmax in deep ... - brown university.” [Online]. Available: https://cs.brown.edu/people/gdk/pubs/tuning_mellowmax_drlw.pdf. [Accessed: 14-Sep-2022]. 

[Soft Mellowmax](https://arxiv.org/pdf/2012.09456.pdf):
Y. Gan, Z. Zhang, and X. Tan, “Stabilizing Q learning via soft Mellowmax operator,” arXiv.org, 18-Dec-2020. [Online]. Available: https://arxiv.org/abs/2012.09456. [Accessed: 13-Sep-2022]. 



