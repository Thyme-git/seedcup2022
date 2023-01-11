# 初赛提交说明

## 评测环境
`python3.10`裸环境，即只有标准库以及
```
pip
setuptools
wheel
```
可以使用`pip`下载第三方库

## 提交说明
- 如果使用的是`python`语言，那么文件夹命名为`submit_py`，使用`zip`压缩，上传`submit_py.zip`。

- 每个程序员都有自己的码风和偏爱的目录结构，但是便于自动评测，选手需要指定程序的入口，在此我们规定程序的入口为`start.sh`，在`start.sh`中选手需要完成：
  - 环境的配置（依赖安装）
  - 代码运行（运行你的主程序） 

    注意：工作目录默认为`start.sh`所在目录，即在与`start.sh`同层级的终端中运行`bash ./start.sh`可以正常运行你的程序。

### 依赖
`python`安装的所有`package`可以通过`pip freeze`导出（当你有多个`python`环境时，请务必确定`pip`为配套的包管理工具），但当你的环境比较杂乱时`pip freeze`会包含许多多余的`package`，评测环境带宽有限，无用的`package`过多时下载速度会慢的心痛... 对此建议使用[pipreqs](https://github.com/bndr/pipreqs)，此工具会遍历工程并记录所有`import`的`package`及其版本。

## 提交示例
```
./
├── README.md
├── config.json
├── prepare.sh         # preparation things
├── requirements.txt   # auto generated packages
├── start.sh           # run scripts with only one click
└── client             # your client and favorite directory structure
    ├── base.py        # common class/enum
    ├── config.py      # read config file
    ├── req.py         # request packet
    ├── resp.py        # response packet
    ├── ui.py          # ui for debug
    └── main.py        # main entry
```

其中`prepare.sh`内容如下：
```bash
tuna="https://pypi.tuna.tsinghua.edu.cn/simple"  # Tsinghua source to speed up the download of the package

pip install pipreqs -i $tuna # install package `pipreqs`
pipreqs --force .  # traverse the working directory to get the dependencies
pip install -r ./requirements.txt -i $tuna # install dependencies

```
需要将`prepare.sh`放置在你的工作目录中。

其中一键脚本`start.sh`内容如下：
```bash
/bin/bash ./prepare.sh
python client/main.py # Maybe you need to change to your main program entry
```
`start.sh`运行了`prepare.sh`安装依赖并且开启客户端，你钟爱的目录结构与示例不同时，请修改`start.sh`保证可以运行到你的主程序。
需要将`start.sh`放置在你的工作目录中。

## 备注
- 建议选手在提交前测试自己的代码、环境、一键脚本能否正常运行并得到预期结果，每一次提交都会减少评测机会。
- 如开发语言不是`python`，请联系组委会同学。
- 评测时会修改随机种子（但保证所有选手都用相同的随机种子）进行多次测试，游戏**胜利**或**平局**则计算分数（`score = blockNum + killNum * 10`），游戏失败记为0分，最终分数取平均。