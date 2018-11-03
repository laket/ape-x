## Replication of Ape-X (Distributed Prioritized Experience Replay)

This repo replicates the results Horgan et al obtained:

[1] [Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933)

Our code is based off of code from OpenAI baselines. The original code and related paper from OpenAI can be found [here](https://github.com/openai/baselines). Their implementation of DQN was modified to use Tensorflow custom ops.

Although Ape-X was originally a distributed algorithm, this implementation was meant to maximize throughput on a single machine. It was optimized for 2 GPUs (data gathering + optimization) but could be modified to use only one. With 2 GPUs and 20~40 CPUs you should be able to achieve human median performance in about 2 hours.

## How to run

clone repo

```
git clone https://github.com/uber-research/ape-x.git
```

create python3 virtual env

```
python3 -m venv env
. env/bin/activate
```

install requirements
```
pip install tensorflow-gpu gym
```

Follow the setup under `gym_tensorflow/README.md` and run `./make` to compile the custom ops.

launch experiment
```
python apex.py --env video_pinball --num-timesteps 1000000000 --logdir=/tmp/agent
```

Monitor your results with tensorboard
```
tensorboard --logdir=/tmp/agent
```

visualize results
```
python demo.py --env video_pinball --logdir=/tmp/agent
```

# メモ
[ALE](https://github.com/openai/atari-py)をセットアップする。
公式のALEではなくOpenAIが出している版を使うのが標準っぽい。
gym_tensorflow直下にcloneする。

```angular2html
sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
git clone https://github.com/openai/atari-py.git
```

ALEをビルドするのだが、gym_tensorflow/atari/README.mdに従ってCMakeListを書き換えてからビルドする。
これでALEのトップにlibale.soが手に入る。
gym_tensorflow/MakefileがPython2前提なのをPython3に書き換える。

```angular2html
cd gym_tensorflow
make
```

これでgym_tensorflow.soが手に入る。次にapex_tensorflow.soをビルドする。



以下のエラーが出た場合はいつもの-D_GLIBCXX_USE_CXX11_ABI=0忘れのため、aleをビルドしなおす。

```
/home/laket/project/ape-x/gym_tensorflow/gym_tensorflow.so: undefined symbol: _ZN12ALEInterface8setFloatERKSsf
```



https://github.com/openai/atari-py.git
