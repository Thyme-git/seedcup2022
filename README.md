# 2022 种子杯赛题
##目录结构
```
./
├── README.md
├── config.json
├── seedcupServer  # seedcup server
├── bot            # baseline model
├── documents
│   ├── 2022种子杯初赛试题.pdf
└── client
    ├── base.py    # commen class/enum
    ├── config.py  # read config file
    ├── req.py     # request packet
    ├── resp.py    # response packet
    ├── ui.py      # ui for debug
    └── main.py    # main entry
```

## 使用说明
默认工作目录在最外层目录即为上图所示的``./``
```bash
# launch server
./seedcupServer # if run into permission denied problem, run `chmod +x server` first

# launch python client
python client/main.py


# launch bot
./bot # if run into permission denied problem, run `chmod +x server` first
```

python客户端提供了玩家手玩的功能，两名玩家可以连接同一台机器手玩游戏体验游戏机制。

## 传输协议
客户端和服务端通信需要遵循一定的协议，为了便于选手debug，采用json序列化及反序列化。在python客户端中已经实现了通信的协议，理论上选手可以直接按照客户端提供的接口即可。

### Reqeust协议
#### 总协议
总的协议体如下：`type`字段为`1`表示`InitReq`，`type`字段为`2`表示`ActionReq`。`data`字段则包含具体的请求。
```json
{
  "type": 1,
  "data": {

  }
}
```
#### InitReq
`Init`请求告知服务端主武器和特殊武器的类型。
```json
{
    "masterWeaponType": 1,
    "slaveWeaponType": 2
}
```
#### ActionReq
`Action`请求告知服务端客户端要进行的具体行动。
```json
{
    "characterID": 1,
    "actionType": 2,
    "actionParam": {}
}
```

### Response协议
总的协议体如下：`type`字段表示`resp`类型，`data`字段表示对应的具体的值。
```json
{
  "type": 1,
  "data": {

  }
}
```
#### ActionResp
`ActionResp`会返回击杀数`kill`，当前得分`score`，以及整个地图信息，选手可以利用这些信息训练模型。
```json
{
    "playerID": 0,
    "frame": 1,
    "color": 1,
    "kill": 1,
    "score": 20,
    "characters": [],
    "map":{
        "blocks": [
            {
                "x": 0,
                "y": 0, 
                "frame" : 1,
                "valid": true,
                "color": 2,
                "objs": [

                ]
            }
        ]
    }
}
```
##### Obj
每个`block`可能含有0个或多个`obj`, `obj`有三种类型:`Character`，`Item`，`SlaveWeapon`。
`Character`为玩家操控的角色，有以下属性
```json
{
    "x": 0,
    "y": 0,
    "playerID": 0,
    "characterID": 0,
    "direction": 1,
    "color": 2,
    "hp": 2,
    "moveCD": 2,
    "moveCDLeft": 2,
    "isAlive": true,
    "isSneaky": true,
    "isGod": false,
    "rebornTimeLeft": 0,
    "godTimeLeft": 0,
    "slaveWeapon":{},
    "masterWeapon": {}
}
```
`Item`为场上的增益`buff`，有以下属性
```json
{
    "buffType": 0
}
```
`SlaveWeapon`为特殊武器，有以下属性
```json
{
    "weaponType": 1,
    "playerID": 1
}
```
#### GameOverResp
当游戏结束时会发送`GameOverResp`，有以下属性
```json
{
    "scores": [20,22],
    "result": 1
}
```



