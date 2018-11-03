'''
Copyright (c) 2018 Uber Technologies, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import tensorflow as tf
import numpy as np

import models
from ops.segment_tree import ShortTermBuffer

from gym_tensorflow.wrappers.base import BaseWrapper

def make_masked_frame(frames, dones, data_format):
    """doneなframesは0、それ以外はもとの値を持つTensor群を返す

    :param list[tf.Tensor] frames: ここのTensorはNCHWっぽい (data_formatに従う)
    :param tuple[tf.Tensor] dones: ここのTensorはframes[i]の0次元目の長さと一致するbool
    :param data_format:
    :return:
    """
    frames = list(frames[:])
    mask = None
    # donesを反転して次元を後ろに4つつける (4,) => (4,1,1,1)
    not_dones = [tf.cast(tf.logical_not(d), frames[0].dtype) if d is not None else None for d in dones]
    not_dones = [tf.expand_dims(d, axis=-1) if d is not None else None  for d in not_dones]
    not_dones = [tf.expand_dims(d, axis=-1) if d is not None else None  for d in not_dones]
    not_dones = [tf.expand_dims(d, axis=-1) if d is not None else None  for d in not_dones]
    for i in np.flip(np.arange(len(frames) - 1), 0):
        if mask is None:
            mask = not_dones[i]
        else:
            mask = mask * not_dones[i]
        frames[i] = tf.image.convert_image_dtype(frames[i] * mask, tf.float32)
    frames[-1] = tf.image.convert_image_dtype(frames[-1], tf.float32)
    if data_format == 'NHWC':
        return tf.concat(frames, axis=-1, name='concat_masked_frames')
    elif data_format == 'NCHW':
        return tf.concat(frames, axis=-3, name='concat_masked_frames')
    else:
        raise NotImplementedError()


class ReplayBufferWrapper(BaseWrapper):
    """行動をBufferに蓄積する環境

    BaseWrapperは環境用のクラス

    利用例 (Prioritizedはこのクラスを継承している)
    PrioritizedReplayBufferWrapper(envs[actor_num], actor_num, actor_fifo, framestack, data_format, multi_step_n=multi_step_n)

    """

    def __init__(self, env, actor_num, queue, num_stacked_frames, data_format):
        """

        :param gym_tensorflow.atari.tf_atari.AtariEnv env: step等の関数を持つ環境 (AtariEnvとかくる)
        :param actor_num:
        :param tf.FIFOQueue queue:
        :param int num_stacked_frames: おそらく状態として何フレームを一括として扱うか
        :param data_format:
        """
        super(ReplayBufferWrapper, self).__init__(env)
        self.queue = queue
        self.actor_num = actor_num
        self.num_stacked_frames = num_stacked_frames
        self.data_format = data_format

        with tf.device('/cpu:0'):
            if data_format == 'NCHW':
                obs_space = env.observation_space[0], env.observation_space[-1], env.observation_space[1], env.observation_space[2]
            else:
                obs_space = env.observation_space
            # 常にnum_stacked_framesをトラックする
            self.buffer = ShortTermBuffer(shapes=[obs_space, (env.batch_size,)], dtypes=[tf.uint8, tf.bool],
                                          framestack=num_stacked_frames, multi_step=0)

    @property
    def observation_space(self):
        return self.env.observation_space[:-1] + (self.env.observation_space[-1] * self.num_stacked_frames, )

    def observation(self, indices=None, reset=False, name=None):
        """現在のstateを返す。ただし、num_stacked_frames分拡張されたobservationを返す

        :param indices: batchの中で一部のものをtrackしている場合かな？ (どこで使っている？)
        :param reset: 未使用
        :param name: 未使用
        :return:
        """
        assert indices is None
        obs = self.env.observation(indices)
        if self.data_format == 'NCHW':
            obs = tf.transpose(obs, (0, 3, 1, 2))

        with tf.device('/cpu:0'):
            _, recent_obs_done = self.buffer.encode_history()

            observations, dones=zip( * recent_obs_done[1 - self.num_stacked_frames:])
            observations += (obs,)
            dones += (None,)

        return make_masked_frame(observations, dones, self.data_format)

    def step(self, action, indices=None, name=None):
        assert indices is None
        sliced_act_obs = self.env.observation(indices)
        if self.data_format == 'NCHW':
            sliced_act_obs = tf.transpose(sliced_act_obs, (0, 3, 1, 2))

        sliced_act_obs = tf.image.convert_image_dtype(sliced_act_obs, tf.uint8)
        assert sliced_act_obs.dtype == tf.uint8

        with tf.device('/cpu:0'):
            _, recent_obs_done = self.buffer.encode_history()

            observations, dones=zip( * recent_obs_done[1 - self.num_stacked_frames:])
            observations += (sliced_act_obs,)
            dones += (None,)

        # 直近の4フレームをstateとしてまとめる
        obs = make_masked_frame(observations, dones, self.data_format)
        with tf.control_dependencies([sliced_act_obs]):
            # 1stepすすめる
            rew, done = self.env.step(action=action, indices=indices, name=name)
            # (入力画像, 完了済み)のペアをShortTermBufferに入れる
            # 遷移後のstateは次のstepなりobservationなりで取る思想っぽい
            update_recent_history = self.buffer.enqueue([sliced_act_obs, done])

            # 観測列をReplayBufferに入れる
            enqueue_op = self.queue.enqueue([obs, sliced_act_obs, rew, done, action, self.actor_num])

            with tf.control_dependencies([update_recent_history[0].op, enqueue_op]):
                return tf.identity(rew), tf.identity(done)


class PrioritizedReplayBufferWrapper(ReplayBufferWrapper):
    """ReplayBuffer (Ape-X 所属)

    呼び出し例
    PrioritizedReplayBufferWrapper(envs[actor_num], actor_num, actor_fifo, framestack, data_format, multi_step_n=multi_step_n)

    """

    def __init__(self, *args, multi_step_n=None, **kwargs):
        super(PrioritizedReplayBufferWrapper, self).__init__(*args, **kwargs)
        self.transition_buffer = None
        self.multi_step_n = multi_step_n

    @classmethod
    def get_buffer_dtypes(cls, multi_step_n, framestack):
        return [tf.uint8, tf.float32, tf.bool, tf.int32, tf.float32, tf.float32] * (multi_step_n + framestack)

    @classmethod
    def get_buffer_shapes(cls, env, multi_step_n, num_stacked_frames, data_format):
        b = (env.batch_size,)
        if data_format == 'NCHW':
            obs_space = env.observation_space[-1], env.observation_space[1], env.observation_space[2]
        else:
            obs_space = env.observation_space[1:]
        shapes = [
            obs_space,  # Image
            (), # Reward
            (), # Done
            (), # Action
            (env.action_space,), # Q Values
            (), # Selected Q Value
        ]
        shapes = [b + s for s in shapes]
        return shapes * (multi_step_n + num_stacked_frames)

    def step(self, action, indices=None, name=None, q_values=None, q_t_selected=None):
        """環境を1stepすすめる

        呼び出し例
        env.step(output_actions, q_values=q_values, q_t_selected=q_t_selected)


        :param tf.Tensor action: 選んだアクション [batch_size]
        :param indices:
        :param name:
        :param tf.Tensor q_values:  各アクションのQ(s,a) [batch_size, num_actions]
        :param tf.Tensor q_t_selected:  選んだアクションの評価値 [batch_size]
        :return:
        """

        assert indices is None
        assert q_values is not None
        assert q_t_selected is not None
        batch_size = self.env.batch_size
        # NHWCの画像がとれる
        sliced_act_obs = self.env.observation(indices)
        if self.data_format == 'NCHW':
            sliced_act_obs = tf.transpose(sliced_act_obs, (0, 3, 1, 2))

        sliced_act_obs = tf.image.convert_image_dtype(sliced_act_obs, tf.uint8)
        assert sliced_act_obs.dtype == tf.uint8

        with tf.device('/cpu:0'):
            _, recent_obs_done = self.buffer.encode_history()

            # 最後のnum_stacked_frames-1分だけrecent_obs_doneからとってくる
            observations, dones=zip( * recent_obs_done[1 - self.num_stacked_frames:])
            # 最新の観測を足す Invadorだと(4,1,84,84)が4つのlist
            observations += (sliced_act_obs,)
            # (4,)のboolが4つのlist
            dones += (None,)

        obs = make_masked_frame(observations, dones, self.data_format)
        with tf.control_dependencies([sliced_act_obs]):
            rew, done = self.env.step(action=action, indices=indices, name=name)
            update_recent_history = self.buffer.enqueue([sliced_act_obs, done])

            # (action前状態, 報酬, 終わったかどうか, 選択したアクション, Q[batch_size,num_action], 選んだアクションの価値[batch_size])
            current_frame = sliced_act_obs, rew, done, action, q_values, q_t_selected
            if self.transition_buffer is None:
                with tf.control_dependencies(None):
                    with tf.device('/cpu:0'):
                        self.transition_buffer = ShortTermBuffer(shapes=[v.get_shape() for v in current_frame], dtypes=[v.dtype for v in current_frame], framestack=self.num_stacked_frames, multi_step=self.multi_step_n)

            # ShortTermBufferに現在の状態を足す
            # historyにはnum_stacked_frame+multi-step分のcurrent_frame列が入る
            is_valid, history = self.transition_buffer.enqueue(current_frame)

            history = [e for t in history for e in t]
            replay_queue_shapes = [(None,) + tuple(a.get_shape()[1:]) for a in history]

            enqueue_op = tf.cond(is_valid, lambda: self.queue.enqueue(history), tf.no_op)

            with tf.control_dependencies([enqueue_op, update_recent_history[0].op]):
                return tf.identity(rew), tf.identity(done)
