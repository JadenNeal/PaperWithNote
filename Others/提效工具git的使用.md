---
title: 提效工具git的使用
date: 2019-12-27 16:19:56
tags: 其他
---
安装就不废话了，各种操作平台安装见官网就行了。
<!--more-->

<div align="center"><img src="https://s2.ax1x.com/2019/12/27/lV4ZcR.png" alt="lV4ZcR.png" border="0" /></div>
<center>图片引用自菜鸟教程</center>

## 基本使用
以windows为例，首先打开git bash（即git的终端）
- 首先创建一个文件夹作为仓库（或者其他的想作为代码库的文件夹）
这里我在D盘创建了一个名为"GitTUT"的文件夹。
- 接着，初始化用户名和邮箱
```git
git config --global user.name "JadenNeal"  # 用户名
git config --global user.email "Ran@example.com"  # 邮箱
# 这里只是一个身份的象征，不需要真实。
```
-  进入之前准备好的文件夹，初始化
```git
git init
```
- 创建新文件
```git 
touch 2.py  # 也可以不用命令行创建
```
- 查看状态
```git 
git status
```
<div align="center"><img src="https://s2.ax1x.com/2019/12/27/lV5VIS.jpg" alt="lV5VIS.jpg" border="0" /></div>
- 将文件加入到暂存区
```git 
git add 2.py  # 单个文件添加
git add .  # 添加全部文件
```
- 提交更改
```git
git commit -m "change 1"  # change 1 是添加的备注说明
```

## 连接到github
- 首先创建一个新的代码库，注意名称不需要和本地仓库名称相同。
- 在这一行提示下：push an exsiting repository from the command line
- 复制第一行到bash中执行
- 再复制第二行到bash中执行（第一次需要输入用户名和密码）
- 最后执行`git push -u origin master`
- 刷新github的页面，发现已经将刚刚创建的2.py上传上去了。
