# Deep Reinforcement Learning using DQN and RNNs
![GitHub Logo](/images/groupInteraction.png)

User policy learning will augment our current offline sequential prediction model with an online update mechanism to enable the model to adapt to changing user behavior during test time (diagram on slide 8)
Offline module extracts features from the Reddit datasets and prepares an initial recommendation
Using a combination of user clicks and user return intervals a DQN (Deep Q-learning) is used to predict the future reward.
Reward is a combination of user + subreddit posts and user activeness, i.e. how often they return.
In online learning, the recommender agent will receive feedback from the user posts which it will use for a minor update step 
Minor update which will compare the reward from exploitation vs exploration.
If the future reward from exploration is higher, the model will be use exploration network Q, otherwise it will stick to the exploitation network. 
In a major update step the recommender system will use all the feedback from the previous user-group interactions and user activity levels (how much the user posts) stored in memory to update the Q network .
