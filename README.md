# Autonomous Navigation Agent: Super Mario Bros

A reinforcement-learning agent that learns to play Super Mario Bros from raw game pixels. It uses Proximal Policy Optimisation (PPO) with a convolutional policy network.

The full walkthrough is in [`AI_Plays_Mario.ipynb`](AI_Plays_Mario.ipynb).

## What the notebook covers

1. **Environment setup:** `gym-super-mario-bros` with the `nes-py` emulator, and the `SIMPLE_MOVEMENT` action set.
2. **Exploration:** random play to check the observation and action spaces.
3. **Preprocessing:** grayscale frames, a dummy vectorised environment, and a stack of 4 frames so the agent can see motion.
4. **Training:** PPO with a `CnnPolicy`, a callback that saves a checkpoint every 50,000 steps, and TensorBoard logs.
5. **Playback:** loading a saved model and running it on the game.

## Skills demonstrated

- Reinforcement learning (PPO, policies, reward-driven training)
- Deep learning with convolutional networks on image input
- Gym-style environment wrappers and vectorised environments
- Experiment management: checkpointing and TensorBoard logging

## Running it

```bash
pip install gym-super-mario-bros nes-py stable-baselines3 gym matplotlib
jupyter notebook AI_Plays_Mario.ipynb
```

## Notes

- The training and checkpoint paths point to an external drive on macOS (`/Volumes/Mario/`). Change `CHECKPOINT_DIR` and `LOG_DIR` to a folder on your machine first.
- Training runs for one million timesteps and takes a long time on a laptop. A saved checkpoint is needed to run the agent without retraining.
