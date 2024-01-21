# Warm up learning scheduler

目前 warm up 结合的 scheduler 通常是 cosine 学习率，因此使用 cosine 学习率作为示例。

首先需要理清 `step` 和 `epoch` 的概念。`step` 是训练一个 `batch size` 的数据，而 `epoch` 则可能包含多个 `step`。

## 按 Epoch 调整学习率

该方法是网上教程比较常见的方法，就是为每个 `epoch` 设置不同的学习率。参考下面代码。

```python
def warm_up_lr(epoch):
    if epoch < args.warm_up_epochs:
        return epoch / args.warm_up_epochs
    else:
        return 0.5 * (math.cos((epoch - args.warm_up_epochs) / (args.max_epochs - args.warm_up_epochs) * math.pi) + 1)
```

这样就能将上述函数传给 `LambdaLR` 学习器作为参数。

```python
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_lr)
```

这样就能使用 `scheduler.step()`来不断更改学习率了。

## 按 Step 调整学习率

事实上，许多大佬使用的 `warm up` 是针对每个 `step` 的。也就是说，为每个 `step` 设置不同的学习率。

这种方法就需要自定义 `scheduler`，其实也就是自定义一个学习率列表，然后每个 `step` 根据列表进行索引即可。

参考 `ConvNext` 中的 `utils.py`。

```python
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    # 从第0个step开始，一直到设置的warm up epoch最后一个step
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule
```

这样就能在模型训练的时候手动修改学习率。

```python
model = Net()  # 生成网络
optimizer_Adam = torch.optim.Adam(model.parameters(), lr=0.1)  # 生成优化器
lr_schedule = cosine_scheduler(...)

for epoch in range(100):  # 假设迭代100个epoch 
    for i, (x, y) in train_loop:
        idx = epoch * len(loader_train) + i
        for params in optimizer_Adam.param_groups:             
            # 遍历Optimizer中的每一组参数           
            params['lr'] = lr_schedule[idx]            
            # params['weight_decay'] = 0.5  # 当然也可以修改其他属性
        ...
```

目前先写这些，如有需要再继续补充细节。
