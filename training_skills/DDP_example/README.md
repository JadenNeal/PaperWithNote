# PyTorch DDP 分布式

本文介绍PyTorch 分布式训练，主要讲解单机多卡情况。

## 1. 单卡训练

### 1-1. 示例模型

首先，创建一个示例模型。

```python
# model.py
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        out = self.extractor(x)
        out = self.squeeze(out)
        out = self.head(out)

        return out
```

### 1-2. 单卡训练伪代码

单卡的情况，训练伪代码如下。

```python
# main.py
device = torch.device(f"cuda:{args.gpu}") 

train_set = MyDataset(...)
train_loader = DataLoader(...)

model = ToyModel().to(device)

...
for i, (x, y) in train_loop:
    x = x.to(device)
    y = y.to(device)

    y_hat = model(x)
    loss = criterion(y_hat, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

...
```

### 1-3. 单卡训练脚本

```bash
CUDA_VISIBLE_DEVICES=1 python main.py
```

## 2. DP多卡训练

### 2-1. DP训练伪代码

`torch.nn.DataParallel`能够实现多卡训练。

与单卡类似，只不过更改了模型包装方式。

```python
...
model = ToyModel().to(device)
model = torch.nn.DataParallel(model)
...
```

### 2-2. DP执行脚本

```bash
CUDA_VISIBLE_DEVICES=1,2 python main.py
```

## 3. DDP多卡训练

### 3-1. 基本概念（单机多卡）

1. `rank`: GPU编号。多机中为机器的编号。
2. `local_rank`: GPU编号。
3. `world_size`: GPU数量。

DDP的大致思想是将`dataset`平均划分为`world_size`份，然后分别训练后汇总结果。可以简单地理解为，一次性执行了`world_size`份训练代码，不过每个gpu上的数据不同。

要实现`DDP`训练，主要修改/增加 四个步骤：

1. 初始化。主要是`dist.init_process_group(backend='nccl')`
2. 设置device。复制model到对应的GPU上。`device = torch.device("cuda", args.gpu)`
3. 包装模型。`model = DDP(model, device_ids=[args.gpu])`
4. 划分数据。`sampler = DataLoader(train_set, sampler=train_sampler, ...)`

### 3-2. 初始化

都是固定的操作，需要注意的是`PyTorch`更新后已经不需要设置`local_rank`了，而是可以直接获取。

```python
# utils.py
def init_distributed_mode(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
    # linux系统一般选'nccl'即可，最合适速度最快
```

### 3-3. 数据准备与模型包装

初始化之后，需要设置`device`，方便后续模型和数据的复制。

```python
device = torch.device("cuda")
```

为了复现实验结果，还需要设置`seed`。这里要注意不能设置相同的`seed`，否则多张GPU会显示出**同态性**，导致性能下降。

```python
seed = 666 + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.deterministic = True
```

然后就是数据集的准备。

```python
train_set = MyDataset(...)
train_sampler = DistributedSampler(train_set, shuffle=True, seed=666)
train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=args.batch_size)

# 要是想要分布式验证的话，验证集也可以这么做。
```

接着就是核心的模型包装。其结果就是将同一个模型分别复制到不同的GPU上。

```python
model = ToyModel() # 实例化模型
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
################################################################################
# 这一步是因为ToyModel()中含有BN层，BN层则需要全局数据计算mean和std
# 而DDP将数据分为了几个部分，若不加上面这句，计算的mean和std仅仅是局部的，而不是全局的。
################################################################################
model = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
# 包装模型
```

### 3-4. DDP训练

其他的设置和单卡类似（损失函数、优化器等），在训练的时候，需要加上下面语句来进行打乱操作。

```python
...

for epoch in range(epochs):
    loader_train.sampler.set_epoch(epoch)  # 打乱train_loader
    train_one_epoch(...)
    val_one_epoch(...)

...
```

这样一来，DDP训练就完成了。

#### 3-4-1. dist.all_reduce()的争论

虽然能够使用上面的代码进行DDP训练，但是，怎么保证多张卡训练得到的model参数是一致的（同一个模型）呢？

**答案是不需要操作**。

有人在复现DDP功能的时候，需要手动实现梯度更新并多卡同步。主要思路是获取各个子模型的参数，汇总梯度并平均，然后更新给所有的卡，使用的就是`dist.all_reduce`函数。

[大致思路](https://zhuanlan.zhihu.com/p/482557067)如下：

```python
def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size
```

但实际上，只要我们使用DDP进行模型包装，后续执行反向传播即可，参见[官方论坛上的讨论](https://discuss.pytorch.org/t/when-will-dist-all-reduce-will-be-called/129918)。

```python
loss.backward()
```

DDP库会自动帮我们汇总梯度，更新后广播给其他卡上的模型，这样就保证了所有卡上的模型是一致的。

#### 3-4-2. dist.all_reduce()的使用

虽然在训练过程中不需要显式调用`dist.all_reduce()`来更新梯度，但倘若想要监控训练过程中的loss，该如何做？

`dist.all_reduce()`的作用就是对多张卡上的张量`t`执行操作（累加，累乘，求最大，求最小），原地修改并广播给其他卡。因此，若要求全局的loss，可以参考下面语句。

```python
def int2bool(v):
    """
    integer to boolean
    """
    if v == 0:
        return False
    else:
        return True


def synchronize_between_processes(t):
    # 注意一定要有device参数
    t = torch.tensor(t, dtype=torch.float64, device='cuda')
    dist.barrier()      # 等待其他GPU进程
    dist.all_reduce(t)  # 汇总，原地修改

    return t.item()

...

model.train()
train_loss = 0.
length = 0

# 由于多个进程会分别执行，因此不加限制的话每个进程都会展示一个tqdm进度条
# 使用`disable`参数对是否展示进行限制。
train_loop = tqdm(enumerate(loader_train, 0), total=len(loader_train), colour='blue', disable=int2bool(rank))
for i, (x, y) in train_loop:
    x = x.to(device)
    y = y.to(device)

    y_hat = model(x)

    loss = criterion(y_hat, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    num = y.size(0)
    train_loss += loss.item() * num
    length += num

# 汇总样本数量
length = synchronize_between_processes(length)  
# 汇总loss
train_loss = synchronize_between_processes(train_loss)

avg_train_loss = train_loss / length  # 全局的平均loss
```

这样一来，就能查看训练过程中的loss指标了。

### 3-5. DDP验证

验证过程与训练过程类似，不同的是验证过程不需要反向传播更新模型，以及很有必要查看验证的loss。

```python
model.eval()
valid_loss = 0.
length = 0
with torch.no_grad():
    val_loop = tqdm(enumerate(loader_val, 0), total=len(loader_val), colour='white', disable=int2bool(rank))
    for i, (x, y) in val_loop:
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        loss = criterion(y_hat, y)

        num = y.size(0)
        valid_loss += loss.item() * num
        length += num

length = synchronize_between_processes(length)
valid_loss = synchronize_between_processes(valid_loss)

avg_val_loss = valid_loss / length
```

### 3-6. DDP测试与推理

测试/推理过程与验证过程不同，其目的是获取模型的预测输出，而不是损失的loss。经过尝试，在大数据集上使用`dist.gather()`速度相当慢，理由是该函数需要预先创建对应张量空间，然后复制数据。`dist.gather`用法可以自查一下。

于是，推理的时候建议使用单卡测试。

因此保存模型的时候注意保存为单卡模型。

```python
# 保存model.module
torch.save(model.module.state_dict(), "best_model.pth")
```

推理过程大致如下：

```python
test_set = MyDataset(...)
# 普通模式创建test_loader
test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=2, shuffle=False) 

ckpt = torch.load("best_model.pth", map_location="cpu")  
# 先load到cpu上比较好，防止model过大爆GPU显存，速度也更快
test_model = ToyModel().to(device)
test_model.load_state_dict(ckpt, strict=False)

inference(...)

```

### 3-7. 执行脚本

假设使用4张卡。

```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 main.py [--script parameters]
```

`--nproc_per_node`要与前面的卡数对应起来。

要是直接使用全部的卡，可以省去前面的限制。

```bash
torchrun --nproc_per_node=8 main.py [--script parameters]
```

## 参考资料

1. [PyTorch分布式训练](https://zhuanlan.zhihu.com/p/76638962)
2. [[原创][深度][PyTorch] DDP系列第一篇：入门教程](https://zhuanlan.zhihu.com/p/178402798)
3. [[原创][深度][PyTorch] DDP系列第二篇：实现原理与源代码解析](https://zhuanlan.zhihu.com/p/187610959)
4. [[原创][深度][PyTorch] DDP系列第三篇：实战与技巧](https://zhuanlan.zhihu.com/p/250471767)
5. [PyTorch官方论坛关于dist.all_reduce的讨论](https://discuss.pytorch.org/t/when-will-dist-all-reduce-will-be-called/129918)
6. [ConvNext训练模板](https://github.com/facebookresearch/ConvNeXt)
7. [github上一个包含evaluate的例子](https://github.com/KaiiZhang/DDP-Tutorial/blob/main/codes/mnist-env.py)
