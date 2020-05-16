import numpy as np
import core
import tensorflow as tf
import gym
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import pybullet_envs


class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    """存储 obs, act, rew, val, log_p"""
    def store(self, obs, act, rew, val, log_p):

        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = log_p
        self.ptr += 1

    """计算GAE-Lambda 优势函数， rewards-to-go"""
    def finish_path(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    """获取所存储的数据，包括：状态:obs, 动作:act, 优势函数:adv, rewards-to-go:ret, 对数概率:logp"""
    def get(self):
        assert self.ptr == self.max_size    # buffer 必须填满
        self.ptr, self.path_start_idx = 0, 0
        # 接下来的两行实现 advantage normalization
        adv_mean, adv_std = core.statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf

def vpg(env_fn, actor_critic=core.ActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, save_freq=10):

    np.random.seed(seed)
    tf.random.set_seed(seed)
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # 创建 actor-critic model
    actor_critic = actor_critic(env.observation_space, env.action_space, **ac_kwargs)   

    # Experience buffer
    buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
    # Adam优化器
    pi_optimizer = tf.optimizers.Adam(learning_rate=pi_lr)
    vf_optimizer = tf.optimizers.Adam(learning_rate=vf_lr)

    def train_vpg(obs_buf, act_buf, adv_buf, ret_buf, logp_buf):
        # 执行一步梯度下降训练policy
        with tf.GradientTape() as pi_tape:
            _, logp = actor_critic.policy([obs_buf, act_buf])
            pi_loss = -tf.reduce_mean(logp * adv_buf)
        pi_grads = pi_tape.gradient(pi_loss, actor_critic.policy.trainable_variables)
        pi_optimizer.apply_gradients(zip(pi_grads, actor_critic.policy.trainable_variables))

        # Info (useful to watch during learning)
        approx_kl = tf.reduce_mean(logp_buf - logp)      # a sample estimate for KL-divergence, easy to compute
        approx_ent = tf.reduce_mean(-logp)               # a sample estimate for entropy, also easy to compute

        # Value 学习
        for _ in range(train_v_iters):
            with tf.GradientTape() as vf_tape:
                v = actor_critic.v_mlp(obs_buf)
                v_loss = tf.reduce_mean((ret_buf - v) ** 2)
            vf_grads = vf_tape.gradient(v_loss, actor_critic.v_mlp.trainable_variables)
            vf_optimizer.apply_gradients(zip(vf_grads, actor_critic.v_mlp.trainable_variables))


    # 用 Checkpoint 管理和加载模型参数
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0),myAwesomeModel=actor_critic)
    manager = tf.train.CheckpointManager(checkpoint, directory='./save_Hopper', max_to_keep=1)

    checkpoint.restore(tf.train.latest_checkpoint('./save_Hopper'))
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    summary_writer = tf.summary.create_file_writer('./tensorboard_Hopper')

    obs, ep_ret, ep_len = env.reset(), 0, 0
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(steps_per_epoch):

            checkpoint.step.assign_add(1)

            obs = tf.reshape(obs,shape=[1, obs.shape[0]])
            a, logp_t, v_t = actor_critic((obs,None))

            next_obs, r, d, _ = env.step(a[0].numpy())
            ep_ret += r
            ep_len += 1

            # 保存obs, a, r, v_t, logp_t
            buf.store(obs, a, r, v_t, logp_t)
            
            # 更新 obs (critical!)
            obs = next_obs

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    obs = tf.reshape(obs,shape=[-1, obs.shape[0]])
                    _, _, last_val = actor_critic((obs,None))
                else:
                    last_val = 0
                buf.finish_path(last_val)
                if terminal:
                    with summary_writer.as_default():  # 希望使用的记录器
                        tf.summary.scalar("summary_ep_ret", ep_ret/ep_len, step=int(checkpoint.step))
                        tf.summary.scalar("all_ep_ret", ep_ret, step=int(checkpoint.step))
                obs, ep_ret, ep_len = env.reset(), 0, 0

        # 打印和保存模型
        print('epoch - %d step - %d' %(epoch, int(checkpoint.step)))
        if epoch % 10 == 0:  # 每隔10个epoch保存一次
            path = manager.save()
            print("model saved to %s" % path)
        # 更新 VPG !
        obs_buf, act_buf, adv_buf, ret_buf, logp_buf = buf.get()
        train_vpg(obs_buf, act_buf, adv_buf, ret_buf, logp_buf)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HopperBulletEnv-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=700)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()
    # tensorflow==2.1.0
    print(tf.__version__)
    vpg(lambda : gym.make(args.env), actor_critic=core.ActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        )
    