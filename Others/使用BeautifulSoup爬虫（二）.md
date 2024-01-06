---
title: 使用BeautifulSoup爬虫（二）
date: 2019-12-27 16:07:53
tags: 爬虫
---
目标网站：[北京地铁](https://www.bjsubway.com/station/zjgls/#)  
使用模块：**re、os、requests、BeautifulSoup**
<!--more-->

老样子，直接步入正题。  
先看下结果：

<div align="center"><img src="https://s2.ax1x.com/2019/12/27/lVg1hR.jpg" alt="test.jpg" border="0"></div>
<center>北京地铁各站点间距离一览</center>

## 载入模块
```python
import requests
from bs4 import BeautifulSoup
import re
import os
```

## 解析网站
```python
ul = 'https://www.bjsubway.com/station/zjgls/#'
response = requests.get(ul)
response.encoding = 'gbk'  # 原始网页编码错误，utf-8也不管用，只能用gbk
html = response.text
# print(html)
soup = BeautifulSoup(html, 'lxml')  # 变成汤汁
```

由于爬取的次数不多，对于速度没有作限制，个人猜测是有次数限制的。至于次数是多少，感兴趣的朋友可以拿自己的IP撞撞枪口[/doge]。保险的方法是加上自己的User Agent，这一点，可以从浏览器找到，以后再讲讲。

## 网站编码问题
爬取的时候遇到一个网站编码的问题。使用requests爬取的时候，Requests 会基于 HTTP 头部对响应的编码作出有根据的推测。即requests会对网站进行自动编码，以保证人能看得懂，所以一般不用再另外编码。但这个网站爬取的结果里的中文却是乱码，将其重新编码为"utf-8"，发现更乱了，于是改成了“gbk”。

对于requests编码问题，参见[官方文档](http://cn.python-requests.org/zh_CN/latest/user/quickstart.html#id3)

## 分析网页源码
我们想要的“菜”是线路名称以及站点信息。分析网址发现，我们的“菜”被装在'<td colspan="5">'，即“td”tag里面。
<div align="center"><img src="https://s2.ax1x.com/2019/12/27/lVglN9.jpg" alt="1号线.jpg" border="0"></div>
<center>一号线示例</center>

但这么找之后发现，线路不全。比如2号线就没找到，再分析2号线的代码：
<div align="center"><img src="https://s2.ax1x.com/2019/12/27/lVgQAJ.jpg" alt="2号线.jpg" border="0"></div>
<center>二号线</center>

那再把colspan = '7'加进去吧，结果发现还少几条线。原来还有colspan = '6' 和colspan = '9'（十号线，仅此一条）。这样一来，重复的代码有点多，写成一个函数：
```python
def get_txt_name():  # 得到线路名称的前一步
    txt_src_name = []
    for i in range(5, 10):
        temp = soup.find_all('td', {'colspan': str(i)})
        txt_src_name += temp
    return txt_src_name
    # 格式如[<td colspan="6">15号线相邻站间距信息统计表</td>, <td colspan="6">昌平线相邻站间距信息统计表</td>]
# print(get_txt_name())  # 测试用
def get_txtuseful_name():  # 得到可用的线路名称
    obj = []
    for each in get_txt_name():
        temp = re.findall(r">(.+?)<", str(each))  # 从>匹配到<(不包含)，若要包含，则先使用re.compile，再search
        obj += temp
    return obj
# print(get_txtuseful_name())  # 测试用
```
得到的就是包含所有线路名称的列表啦。

## 站点信息
同样地，分析网页发现所有地站点信息都被存储在“<tbody>”这个tag中，于是同样地进行查找并筛选：
```python
Stationinfo = soup.find_all('tbody')  # 汤汁是ResultSet，即结果集
def get_stationinfo():
    obj = []
    for each in Stationinfo:
        temp = re.findall(r">(.+?)<", str(each))   # 正则匹配，str格式
        obj += temp
    return obj
# print(get_stationinfo())
```

到了tbody之后，剩下的数据就没有tag可用了，只能用正则表达式来匹配。查找">xxxx<"的"xxx"即为站点信息。

还有一点要注意，要把soup中的ResultSet转换成str格式才能调用正则表达式。 

## 写入文件
```python
station_list = get_stationinfo()
# print(station_list)
os.makedirs('./线路图/', exist_ok=True)
with open('./线路图/test.txt', 'w') as f:  # 不能是wb，编码有问题，或者str转换成byte
    for line in station_list:
        if line == '上行/下行':
            f.write(line + '\n')
        else:
            f.write(line + '      ')  # 多来几个空格显得好看一点
```

这里还是需要再提一下**编码**问题，前面是str格式的，如果写入文件的格式为'b'，即二进制，则会报错。解决的方法是第一像代码那样仅仅是'w'模式 ；或者在此之前将station_list转换成byte格式，那么使用'wb'模式就不会有问题了。