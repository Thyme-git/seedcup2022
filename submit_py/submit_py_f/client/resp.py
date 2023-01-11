from base import *
from typing import Union, List


class Weapon(JsonBase):
    def __init__(
        self,
        weaponType: Union[SlaveWeaponType, MasterWeaponType] = SlaveWeaponType.Kiwi,
        attackCD: int = 0,
        attackCDLeft: int = 1,
    ) -> None:
        super().__init__()
        self.weaponType = weaponType
        self.attackCD = attackCD
        self.attackCDLeft = attackCDLeft


class MasterWeapon(Weapon):
    def __init__(
        self,
        weaponType: MasterWeaponType = MasterWeaponType.PolyWatermelon,
        attackCD: int = 0,
        attackCDLeft: int = 1,
    ) -> None:
        super().__init__(weaponType, attackCD, attackCDLeft)


class SlaveWeapon(Weapon):
    def __init__(
        self,
        weaponType: SlaveWeaponType = SlaveWeaponType.Kiwi,
        attackCD: int = 0,
        attackCDLeft: int = 1,
    ) -> None:
        super().__init__(weaponType, attackCD, attackCDLeft)


class Character(JsonBase):
    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        playerID: int = 0,
        characterID: int = 0,
        direction: Direction = Direction.Above,
        color: ColorType = ColorType.White,
        hp: int = 0,
        moveCD: int = 0,
        moveCDLeft: int = 0,
        isAlive: bool = True,
        isSneaky: bool = True,
        isGod: bool = False,
        rebornTimeLeft: int = 0,
        godTimeLeft: int = 0,
        slaveWeapon: SlaveWeapon = SlaveWeapon(),
        masterWeapon: MasterWeapon = MasterWeapon(),
    ) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.playerID = playerID
        self.characterID = characterID
        self.direction = direction
        self.color = color
        self.hp = hp
        self.moveCD = moveCD
        self.moveCDLeft = moveCDLeft
        self.isAlive = isAlive
        self.isSneaky = isSneaky
        self.isGod = isGod
        self.rebornTimeLeft = rebornTimeLeft
        self.godTimeLeft = godTimeLeft
        self.slaveWeapon = slaveWeapon
        self.masterWeapon = masterWeapon


class BuffType(JsonIntEnum):
    BuffSpeed = 1
    BuffHp = 2


class Item(JsonBase):
    def __init__(self, buffType: BuffType = BuffType.BuffHp) -> None:
        super().__init__()
        self.buffType = buffType


class SlaveWeapon(JsonBase):
    def __init__(
        self, weaponType: SlaveWeaponType = SlaveWeaponType.Kiwi, playerID: int = 0
    ) -> None:
        super().__init__()
        self.weaponType = weaponType
        self.playerID = playerID


class Obj(JsonBase):
    def __init__(
        self,
        type: ObjType = ObjType.Null,
        status: Union[None, Character, Item, SlaveWeapon] = None,
    ) -> None:
        super().__init__()
        self.type = type
        self.status = status

    def from_json(self, j: str):
        d = json.loads(j)
        self.type = self.type.from_json(d.pop("type"))
        status = d.pop("status")
        if self.type == ObjType.Character:
            self.status = Character().from_json(json.dumps(status))
        elif self.type == ObjType.Item:
            self.status = Item().from_json(json.dumps(status))
        elif self.type == ObjType.SlaveWeapon:
            self.status = SlaveWeapon().from_json(json.dumps(status))

        return self


class Block(JsonBase):
    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        frame: int = 0,
        valid: bool = True,
        color: ColorType = ColorType.White,
        objs: List[Obj] = [],
    ) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.frame = frame
        self.valid = valid
        self.color = color
        self.objs = objs

    def from_json(self, j: str):
        d = json.loads(j)
        for key, value in d.items():
            if key in self.__dict__:
                if key == "objs":
                    self.objs = [Obj().from_json(json.dumps(v)) for v in value]
                elif hasattr(self.__dict__[key], "from_json"):
                    setattr(self, key, self.__dict__[key].from_json(json.dumps(value)))
                else:
                    setattr(self, key, value)
        return self


class Map(JsonBase):
    def __init__(self, blocks: List[Block] = []) -> None:
        super().__init__()
        self.blocks = blocks

    def from_json(self, j: str):
        d = json.loads(j)
        for key, value in d.items():
            if key in self.__dict__:
                if key == "blocks":
                    self.blocks = [Block().from_json(json.dumps(v)) for v in value]
                elif hasattr(self.__dict__[key], "from_json"):
                    setattr(self, key, self.__dict__[key].from_json(json.dumps(value)))
                else:
                    setattr(self, key, value)
        return self


class ActionResp(JsonBase):
    def __init__(
        self,
        playerID: int = 0,
        frame: int = 0,
        color: ColorType = ColorType.White,
        kill: int = 0,
        score: int = 0,
        characters: List[Character] = [],
        map: Map = Map(),
    ) -> None:
        super().__init__()
        self.playerID = playerID
        self.frame = frame
        self.color = color
        self.kill = kill
        self.score = score
        self.characters = characters
        self.map = map

    def from_json(self, j: str):
        d = json.loads(j)
        for key, value in d.items():
            if key in self.__dict__:
                # if key == "characters":
                #     self.characters = [Character().from_json(json.dumps(v)) for v in value]
                if hasattr(self.__dict__[key], "from_json"):
                    setattr(self, key, self.__dict__[key].from_json(json.dumps(value)))
                else:
                    setattr(self, key, value)
        # buggy
        value = d.pop("characters")
        self.characters = [Character().from_json(json.dumps(v)) for v in value]
        return self


class GameOverResp(JsonBase):
    def __init__(
        self, result: ResultType = ResultType.Tie, scores: List[int] = []
    ) -> None:
        super().__init__()
        self.scores = scores
        self.result = result


class PacketResp(JsonBase):
    def __init__(
        self,
        type: PacketType = PacketType.ActionResp,
        data: Union[ActionResp, GameOverResp] = ActionResp(),
    ) -> None:
        super().__init__()
        self.type = type
        self.data = data

    def from_json(self, j: str):
        d = json.loads(j)
        self.type = self.type.from_json(d.pop("type"))
        data = d.pop("data")
        if self.type == PacketType.ActionResp:
            self.data = ActionResp().from_json(json.dumps(data))
        elif self.type == PacketType.GameOver:
            self.data = GameOverResp().from_json(json.dumps(data))

        return self


if __name__ == "__main__":
    gameoverResp = GameOverResp([10, 20])
    gameoverPktResp = PacketResp(PacketType.GameOver, gameoverResp)
    print(gameoverPktResp.to_json())
    s = r'{"type": 4, "data": {"scores": [200,1000]}}'
    print(gameoverPktResp.from_json(s))

    actionResp = ActionResp(
        10,
        2,
        ColorType.Blue,
        [
            Character(
                10,
                2,
                ColorType.Blue,
                100,
                20,
                Weapon(SlaveWeaponType.Kiwi, 10, 20),
                Weapon(MasterWeaponType.PolyWatermelon, 10, 20),
            )
        ],
        Map(
            [
                Block(
                    20,
                    -10,
                    ColorType.Red,
                    [Item(BuffType.BuffSpeed), SlaveWeapon(SlaveWeaponType.Cactus, 2)],
                )
            ]
        ),
    )
    print(actionResp)
    s = r'{"id": 20, "frame": 30, "color": 2, "characters": [{"playerID": 10, "characterID": 2, "color": 2, "hp": 100, "speed": 20, "slaveWeapon": {"weaponType": 1, "attackCD": 10, "attackCDLeft": 20}, "masterWeapon": {"weaponType": 2, "attackCD": 10, "attackCDLeft": 20}}], "map": {"blocks": [{"x": 20, "y": -10, "color": 2, "objs": [{"buffType": 1}, {"weaponType": 2, "playerID": 2}]}]}}'
    print(actionResp.from_json(s))
