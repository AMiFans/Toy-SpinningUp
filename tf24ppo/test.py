import numpy as np
import core
import tensorflow as tf
import gym
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import pybullet_envs
import time

def main():
    env = gym.make('HopperBulletEnv-v0')
    env.render(mode="human")
    obs = env.reset()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # 创建 actor-critic model
    actor_critic = core.ActorCritic(env.observation_space, env.action_space)
    # 实例化Checkpoint，设置恢复对象为新建立的模型actor_critic
    checkpoint = tf.train.Checkpoint(myAwesomeModel=actor_critic)      
    checkpoint.restore(tf.train.latest_checkpoint('./save_Hopper'))    # 从文件恢复模型参数

    while 1:
        time.sleep(1. / 240.)
        obs = tf.reshape(obs,shape=[1, obs.shape[0]])
        pi, _ = actor_critic.policy((obs, None))
        obs, r, done, _ = env.step(pi[0].numpy())
        if done:
            observation = env.reset()

if __name__ == '__main__':
    main()