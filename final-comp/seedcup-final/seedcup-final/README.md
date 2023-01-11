# 2022 种子杯赛题
##目录结构
```
./
├── README.md
├── config.json
├── seedcupServer  # seedcup server
├── documents
│   ├── 2022种子杯赛题.pdf
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

# launch another python client
python client/main.py
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

## 客户端主要更改
1. client/resp.py
    ```git diff
    +++ "release-final/client/resp.py"
    @@ -45,6 +45,8 @@ class Character(JsonBase):
            direction: Direction = Direction.Above,
            color: ColorType = ColorType.White,
            hp: int = 0,
    +        hideCD: int = 0,
    +        hideCDLeft: int = 0,
            moveCD: int = 0,
            moveCDLeft: int = 0,
            isAlive: bool = True,
    @@ -63,6 +65,8 @@ class Character(JsonBase):
            self.direction = direction
            self.color = color
            self.hp = hp
    +        self.hideCD = hideCD
    +        self.hideCDLeft = hideCDLeft
            self.moveCD = moveCD
            self.moveCDLeft = moveCDLeft
            self.isAlive = isAlive
    ```
2. client/ui.py
    ```git diff
    +++ "release-final/client/ui.py"
    @@ -11,13 +11,15 @@ color2emoji = {
        ColorType.Black: Emoji.BlackBrick,
    }

    -playerID2emoji = {
    -    0: Emoji.Character1,
    -    1: Emoji.Character2,
    -    2: Emoji.Character3,
    -    3: Emoji.Character4,
    +playerIDCharacterID2emoji = {
    +    0: {
    +        0: Emoji.Character1,
    +        1: Emoji.Character2,
    +    },
    +    1: {0: Emoji.Character3, 1: Emoji.Character4},
    }

    +
    item2emoji = {BuffType.BuffHp: Emoji.BuffHp, BuffType.BuffSpeed: Emoji.BuffSpeed}

    slave2emoji = {
    @@ -32,6 +34,7 @@ class BlockUI(object):
            x: int,
            y: int,
            color: ColorType = ColorType.White,
    +        frame: int = 0,
            valid: bool = True,
            obj: ObjType = ObjType.Null,
            objData: Union[None, Character, Item, SlaveWeapon] = None,
    @@ -53,17 +56,18 @@ class BlockUI(object):
            self.x = x
            self.y = y
            self.color = color
    +        self.frame = frame
            self.valid = valid
            self.obj = obj
            self.data = objData

    -    def get_emoji(self):
    +    def get_emoji(self, frame):
            """Get emoji according to predetermined priority."""

            def _get_emoji(emoji: Emoji):
                return emoji._value_

    -        if self.valid:
    +        if self.valid and frame == self.frame:
                if self.obj == ObjType.Null:
                    assert isinstance(self.color, ColorType)
                    return _get_emoji(color2emoji[self.color])
    @@ -71,13 +75,17 @@ class BlockUI(object):
                    assert isinstance(self.data, Character)
                    if not self.data.isAlive:
                        return _get_emoji(Emoji.CharacterDead)
    -                return _get_emoji(playerID2emoji[self.data.playerID])
    +                return _get_emoji(
    +                    playerIDCharacterID2emoji[self.data.playerID][self.data.characterID]
    +                )
                elif self.obj == ObjType.Item:
                    assert isinstance(self.data, Item)
                    return _get_emoji(item2emoji[self.data.buffType])
                elif self.obj == ObjType.SlaveWeapon:
                    assert isinstance(self.data, SlaveWeapon)
                    return _get_emoji(slave2emoji[self.data.weaponType])
    +        elif self.valid and self.frame < frame:
    +            return _get_emoji(Emoji.Mosaic)
            else:
                return _get_emoji(Emoji.ObstacleBrick)

    @@ -105,12 +113,20 @@ class UI(object):
            self._characters = characters
            self._score = score
            self._kill = kill
    +        self._frame = 0

        def display(self):

            print(
    -            f"playerID: {self.playerID}, color: {color2emoji[self.color].emoji()}, characterNum: {len(self.characters)}, character: {playerID2emoji[self.playerID].emoji()}, score: {self.score}, killNum: {self.kill}"
    +            f"playerID: {self.playerID}, color: {color2emoji[self.color].emoji()}, characterNum: {len(self.characters)}, characters: ",
    +            end="",
            )
    +        for character in self.characters:
    +            print(
    +                f"{playerIDCharacterID2emoji[self.playerID][character.characterID].emoji()}",
    +                end=" ",
    +            )
    +        print(f"blockNum: {self.score}, killNum: {self.kill}")

            for character in self._characters:
                print(f"characterState: {character}")
    @@ -118,9 +134,18 @@ class UI(object):
            for x in range(self.mapSize):
                print(" " * (self.mapSize - x - 1) * 2, end="")
                for y in range(self.mapSize):
    -                print(self._blocks[x][y].get_emoji(), end="  ")
    +                print(self._blocks[x][y].get_emoji(self.frame), end="  ")
                print("\n")

    +    @property
    +    def frame(self):
    +        return self._frame
    +
    +    @frame.setter
    +    def frame(self, frame):
    +        if frame >= self._frame:
    +            self._frame = frame
    +
        @property
        def playerID(self):
            return self._playerID
    @@ -144,6 +169,7 @@ class UI(object):
                    "y": int,
                    "color": ColorType,
                    "valid": bool,
    +                "frame": int,
                    "obj": ObjType,
                    "objData": data
                }

    ```
3. 
    ```git diff
    +++ "release-final/config.json"
    @@ -1,14 +1,14 @@
    {
        "Host": "0.0.0.0",
        "Port": 9999,
    -    "MapSize": 16,
    +    "MapSize": 24,
        "LoggerName": "seedcup2022",
        "LogDir": "log",
        "ServerMaxConnectionNum": 10,
        "EpollMaxEventsNum": 100,
        "EpollTimeout": 10,
    -    "TimerInitialValue": 200,
    -    "TimerIntervalValue": 200,
    +    "TimerInitialValue": 400,
    +    "TimerIntervalValue": 400,
        "PolyWatermelonAttackDemage": 30,
        "PolyWatermelonAttackCD": 3,
        "DurianAttackDemage": 50,
    @@ -21,18 +21,19 @@
        "CharacterMaxMovingCd": 4,
        "CharacterMinMovingCd": 1,
        "CharacterInitialMovingCD": 4,
    +    "CharacterHidingCD": 1,
        "CharacterKiwiCD": 30,
        "CharacterCactusCD": 50,
        "CharacterDeadTime": 80,
        "CharacterGodTime": 16,
        "BuffSpeedCDData": -1,
        "BuffHPData": 30,
    -    "BuffSpeedProbability": 0.02,
    -    "BuffHPProbability": 0.02,
    -    "ViewRadius": 4,
    +    "BuffSpeedProbability": 0.03,
    +    "BuffHPProbability": 0.04,
    +    "ViewRadius": 5,
        "PlayerNumber": 2,
    -    "CharacterNumber": 1,
    -    "PlayerBornPosition": [[7, -1], [1, -7]],
    ```
```git diff
+++ "b/2022\347\247\215\345\255\220\346\235\257/release-final/client/main.py"
@@ -55,7 +55,7 @@ class Client(object):
     Usage:
         >>> with Client() as client: # create a socket according to config file
         >>>     client.connect()     # connect to remote
-        >>>
+        >>>
     """
     def __init__(self) -> None:
         self.config = config
@@ -161,6 +161,7 @@ def refreshUI(ui: UI, packet: PacketResp):
         ui.characters = data.characters
         ui.score = data.score
         ui.kill = data.kill
+        ui.frame = data.frame

         for block in data.map.blocks:
             if len(block.objs):
@@ -169,6 +170,7 @@ def refreshUI(ui: UI, packet: PacketResp):
                     "y": block.y,
                     "color": block.color,
                     "valid": block.valid,
+                    "frame": block.frame,
                     "obj": block.objs[-1].type,
                     "data": block.objs[-1].status,
                 }
@@ -178,6 +180,7 @@ def refreshUI(ui: UI, packet: PacketResp):
                     "y": block.y,
                     "color": block.color,
                     "valid": block.valid,
+                    "frame": block.frame,
                     "obj": ObjType.Null,
                 }
     subprocess.run(["clear"])
@@ -192,7 +195,9 @@ def recvAndRefresh(ui: UI, client: Client):

     if resp.type == PacketType.ActionResp:
         if len(resp.data.characters) and not gContext["gameBeginFlag"]:
-            gContext["characterID"] = resp.data.characters[-1].characterID
+            gContext["characterID"] = [
+                character.characterID for character in resp.data.characters
+            ]
             gContext["playerID"] = resp.data.playerID
             gContext["gameBeginFlag"] = True

@@ -228,7 +233,7 @@ def main():
     with Client() as client:
         client.connect()

-        initPacket = PacketReq(PacketType.InitReq, cliGetInitReq())
+        initPacket = PacketReq(PacketType.InitReq, [cliGetInitReq(), cliGetInitReq()])
         client.send(initPacket)
         print(gContext["prompt"])

@@ -247,12 +252,13 @@ def main():
             sleep(0.1)

         # IO thread accepts user input and sends requests
-        while not gContext["gameOverFlag"]:
-            if gContext["characterID"] is None:
+        while gContext["gameOverFlag"] is False:
+            if not gContext["characterID"]:
                 continue
-            if action := cliGetActionReq(gContext["characterID"]):
-                actionPacket = PacketReq(PacketType.ActionReq, action)
-                client.send(actionPacket)
+            for characterID in gContext["characterID"]:
+                if action := cliGetActionReq(characterID):
+                    actionPacket = PacketReq(PacketType.ActionReq, action)
+                    client.send(actionPacket)

         # gracefully shutdown
         t.join()
```

## 评测说明
- 评测于14 - 16日开放，在评测之前请准备就绪，没准备就绪将会被判负。
- 每天9点、12点、15点、18点、21点开始比赛，每场比赛开5轮，每轮每支队伍和其他队伍各进行**一局**比赛，**每局比赛**得分规则和初赛规则相同，失败记0分，胜利按照*格子数+击杀数✖️10计算*。
- 每位选手**每场比赛**的得分为**这场比赛中所有对局**得分之和，选手每天的得分为这天内5场比赛的平均，选手最终得分为三天得分的加权值，权重分别为0.2，0.3，0.5。
- 分数算法伪代码如下
```python
totalScore = 0
weights = [0.2, 0.3, 0.5]
for weight, everyday:
    everyDayScore = 0
    for everyRound:
        roundScore = 0
        for group in groups:
            groupScore = 0
            if group != myGroup:
                groupScore = pk(group, myGroup)
            roundScore += groupScore
        everyDayScore += roundScore
    totalScore += weight * everyDayScore
```
- 评测机总会使用最新的代码提交进行评测

