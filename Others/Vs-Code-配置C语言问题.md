# Vs Code 配置C语言问题

**写在最前，本篇文章是针对Windows系统上gdb调试失败问题的，有关完整配置教程，直接参见末尾的参考资料。**

# Vs code配置C语言

最近需要用到C，但是VS太大了，对付一般小项目有点大材小用，所以请出小巧的Vs Code，自带markdown书写，插件居多，很好用就是了。  
那么，配置C/C++该怎么搞呢？  

1. 下载vs code
2. 下载MinGW
3. 配置json文件
4. 代码测试

## 下载vs code

这一步无需多言，打开必应搜**vs code**即可（当然用百度也不拦着，就是需要找一下官网）。
具体网址：https://code.visualstudio.com/

## 下载MinGW

从[这里](https://osdn.net/projects/mingw/downloads/68260/mingw-get-setup.exe/)下载。然后点击安装，路径可选。  
**安装完成之后，将bin目录加入到环境变量**。

<div align="center"><img src="https://s2.ax1x.com/2019/12/27/lVwG1e.png" alt="env_var.png" border="0" heigh="60%" width="60%"></div>  

但值得注意的是，**这一步后来经过检查是失败的**，见下面`launch.json`的配置。因此使用的是dev c++中的MinGW，直接下载[dev c++](https://sourceforge.net/projects/orwelldevcpp/)即可。相应的MinGW的bin目录也应添加到环境变量中。  

## 配置json文件

其实没必要新建c工程，再依次生成json，再进行修改，而是直接新建相应的json文件即可。  
有三个json文件：

- c_cpp_properties.json
- launch.json
- tasks.json

然后是具体配置：

- c_cpp_properties.json

```json
{
    "configurations": [
        {
            "name": "Win32",
            "includePath": [
                "${workspaceFolder}/**"
            ],
            "defines": [
                "_DEBUG",
                "UNICODE",
                "_UNICODE"
            ],
            "windowsSdkVersion": "8.1",
            "compilerPath": "D:\\MinGW\\bin\\gcc.exe",  //MinGW的bin目录下的gcc.exe路径
            "cStandard": "c11",
            "cppStandard": "c++17"
        }
    ],
    "version": 4
}
```

- launch.json

```json
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(gdb) Launch",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/${fileBasenameNoExtension}.exe",// 被调试程序
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": true,
            "MIMode": "gdb",
            "miDebuggerPath": "D:\\DevC++\\Dev-Cpp\\MinGW64\\bin\\gdb.exe",  // 自己电脑的gdb
            "preLaunchTask": "echo",                      // 在调试前需要执行的任务名称
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ]
        }
    ]
}
```

诶？到这一步，里面的`miDebuggerPath`中的`dev c++`从哪冒出来的呢？  
因为到后面，本人发现只要是从官网下载下来的MinGW，写成该路径，断点调试后总会跳出下图所示的错误。
<div align="center"><img src="https://s2.ax1x.com/2019/12/27/lVwQk6.png" alt="error.png" border="0" heigh="60%" width="60%"></div>

经过一番搜索，在[stackoverflow](https://stackoverflow.com/questions/47639685/gdb-error-not-in-executable-format-file-format-not-recognized)上找到了答案，这里为了方便大家，截图解决方案。
<div align="center"><img src="https://s2.ax1x.com/2019/12/27/lVwltK.png" alt="solution.png" border="0" heigh="60%" width="60%"></div>

意思是gcc版本不对，必须使用64位的。  
那么用官网的为什么不行呢？额，在源程序添加断点后，在`Terminal`中输入`gdb test.c`，得到以下提示：
<div align="center"><img src="https://s2.ax1x.com/2019/12/27/lVw8pD.png" alt="mingw32.png" border="0" heigh="60%" width="60%"></div>

原来自带的是32位的，问题就是出在这里，**但我目前确实没找到用哪个才是64位的**  
**所以换个思路，不就是需要MinGW吗？除了官网还有许多替代品，因此还有许多可选选项**  
正好本人电脑上有dev c++，于是直接用的自带的gcc编译器，结果通过了。  
有大佬能解答下最好。

- tasks.json

```json
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "echo",
            "type": "shell",
            "command": "gcc",  //c语言用gcc，c++用g++
            "args": [
                "-g",
                "${file}",
                "-o",
                "${fileBasenameNoExtension}.exe"
            ],
            "problemMatcher": [
                "$gcc"
            ]
        }
    ]
}
```
至此，json文件就配置完成了。

## 代码测试

输入以下代码：

```c
#include <stdio.h>

int main()
{
    printf("This is a test!");
    int i = 0;
    int j = 1;
    int k;
    k = i + j;
    printf("\n k");
    getchar();      // 防止一闪而过
    return 0;
}
```

结果截图：

<div align="center"><img src="https://s2.ax1x.com/2019/12/27/lVwJ6H.png" alt="test.png" border="0" heigh="60%" width="60%"></div>

## 额外的说明

多啰嗦几句。  
其实，单纯编译运行c程序的话只需要`gcc`就可以；同理，单纯编译运行C++程序的话只需要`g++`就可以了。`gdb`**只是为了调试用**。  

于是有人就会问，可是vs code只有调试按钮啊，如果安装了官网下载的MinGW，只要按了`F5`，就会`unable to start debugging ···`，怎么直接运行呢？  
很简单，用最原始的命令行解决。

### 命令行编译运行c程序

首先，新建一个C程序，命名为`test.c`，然后编写。  
接着，在`Terminal`中输入`gcc test.c -o test`，意思是编译`test.c`文件，生成`test.exe`。  
然后，再在终端输入`test`，即可看到输出。
<div align="center"><img src="https://s2.ax1x.com/2019/12/27/lVwYXd.png" alt="example.png" border="0" heigh="60%" width="60%"></div>

## 总结

vs code配置c/c++的方法基本一致，都是下载完之后，将其添加到系统环境变量，再配置相关的json文件即可。本人出问题的时候搜索了大量的教程，也遇到了按照教程配置结果出问题的朋友。比如

- 错误，退出代码为1
- 任务被重用（讲道理，这个不算错误，配置没问题的话是可以正常运行的）
- 错误，退出代码为2
- 错误，启动调试失败（unable to start debugging...）
- ...

个人建议还是搜索错误的关键字，有条件就google，没条件就Bing，百度作最后的备选，相信可以在各种博客、github、代码社区找到相应的问题解决方案。  
这里本人遇到的是启动调试失败，然后使用dev c++的MinGW就成功了。

## 参考资料

1. [暮无雪代码博客](https://www.520mwx.com/view/32843)
2. [stackoverflow上关于调试失败问题的解答](https://stackoverflow.com/questions/47639685/gdb-error-not-in-executable-format-file-format-not-recognized)
