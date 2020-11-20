# poisoning_federated_learning
This code includes experiments for paper "How to Backdoor Federated Learning" (https://arxiv.org/abs/1807.00459)

All experiments are done using Python 3.6 and PyTorch 1.0.

```mkdir saved_models```

```python training.py --params utils/params.yaml```


# visdom
```python -m visdom.server```

http://localhost:8097/env/main

# Enviroments

```export PATH=~/anaconda3/bin:$PATH```

```source activate py36```

```git clone git@github.com:ccfasm/poison_mnist.git```

```git pull git@github.com:ccfasm/poison_mnist.git```

# Cached

```git rm -r --cached .  #清除缓存```

```git add . #重新trace file```

```git commit -m "update .gitignore" #提交和注释```

```git push origin master #可选，如果需要同步到remote上的话```

# Git放弃修改，强制覆盖本地代码

```git fetch --all```

```git reset --hard origin/master```

```git pull```