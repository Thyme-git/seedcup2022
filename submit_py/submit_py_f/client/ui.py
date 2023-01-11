from base import *
from resp import *
from config import config
from functools import reduce

color2emoji = {
    ColorType.White: Emoji.WhiteBrick,
    ColorType.Red: Emoji.RedBrick,
    ColorType.Green: Emoji.GreenBrick,
    ColorType.Blue: Emoji.BlueBrick,
    ColorType.Black: Emoji.BlackBrick,
}

playerID2emoji = {
    0: Emoji.Character1,
    1: Emoji.Character2,
    2: Emoji.Character3,
    3: Emoji.Character4,
}

item2emoji = {BuffType.BuffHp: Emoji.BuffHp, BuffType.BuffSpeed: Emoji.BuffSpeed}

slave2emoji = {
    SlaveWeaponType.Kiwi: Emoji.SlaveWeaponKiwi,
    SlaveWeaponType.Cactus: Emoji.SlaveWeaponCactus,
}


class BlockUI(object):
    def __init__(
        self,
        x: int,
        y: int,
        color: ColorType = ColorType.White,
        valid: bool = True,
        obj: ObjType = ObjType.Null,
        objData: Union[None, Character, Item, SlaveWeapon] = None,
    ) -> None:
        """The block class used to display.

        Args:
            x (int): x coordinate.
            y (int): y coordinate.
            color (ColorType, optional): Defaults to ColorType.White.
            valid (bool, optional): The block is an obstacle or not.If true, it is not an obstacle, otherwise it is.Defaults to True.
            obj (ObjType, optional): Object on the block. Possible situations are character, item, slaveweapon and no object.Defaults to ObjType.Null.
            objData (_type_, optional): Supplementary information of obj.
                                        If obj is Null, then objData will be ignored.
                                        If obj is type Character, then objdata should be Character instance.
                                        If obj is type Item, then objdata should be Item instance.
                                        If obj is type SlaveWeapon, then objdata should be SlaveWeapon instance.
        """
        self.x = x
        self.y = y
        self.color = color
        self.valid = valid
        self.obj = obj
        self.data = objData

    def get_emoji(self):
        """Get emoji according to predetermined priority."""

        def _get_emoji(emoji: Emoji):
            return emoji._value_

        if self.valid:
            if self.obj == ObjType.Null:
                assert isinstance(self.color, ColorType)
                return _get_emoji(color2emoji[self.color])
            elif self.obj == ObjType.Character:
                assert isinstance(self.data, Character)
                if not self.data.isAlive:
                    return _get_emoji(Emoji.CharacterDead)
                return _get_emoji(playerID2emoji[self.data.playerID])
            elif self.obj == ObjType.Item:
                assert isinstance(self.data, Item)
                return _get_emoji(item2emoji[self.data.buffType])
            elif self.obj == ObjType.SlaveWeapon:
                assert isinstance(self.data, SlaveWeapon)
                return _get_emoji(slave2emoji[self.data.weaponType])
        else:
            return _get_emoji(Emoji.ObstacleBrick)

    def __str__(self) -> str:
        return f"x:{self.x}, y:{self.y}, color:{self.color}, valid: {self.valid}, obj:{self.obj}, data:{self.data}"


class UI(object):
    def __init__(
        self,
        playerID: int = 0,
        color: ColorType = ColorType.White,
        characters: List[Character] = [],
        score: int = 0,
        kill: int = 0,
    ) -> None:
        self.mapSize = config.get("MapSize")
        self._blocks = [
            [BlockUI(x=x, y=-y) for y in range(self.mapSize)]
            for x in range(self.mapSize)
        ]

        self._playerID = playerID
        self._color = color
        self._characters = characters
        self._score = score
        self._kill = kill

    def display(self):

        print(
            f"playerID: {self.playerID}, color: {color2emoji[self.color].emoji()}, characterNum: {len(self.characters)}, character: {playerID2emoji[self.playerID].emoji()}, score: {self.score}, killNum: {self.kill}"
        )

        for character in self._characters:
            print(f"characterState: {character}")
        print("\n")
        for x in range(self.mapSize):
            print(" " * (self.mapSize - x - 1) * 2, end="")
            for y in range(self.mapSize):
                print(self._blocks[x][y].get_emoji(), end="  ")
            print("\n")

    @property
    def playerID(self):
        return self._playerID

    @playerID.setter
    def playerID(self, playerID):
        if playerID > 0:
            self._playerID = playerID

    @property
    def block(self):
        return self._blocks

    @block.setter
    def block(self, kwargs: dict):
        """Set block attributes.

        supported key value pair:
            {
                "x": int,
                "y": int,
                "color": ColorType,
                "valid": bool,
                "obj": ObjType,
                "objData": data
            }

        """
        block = self._blocks[kwargs.pop("x")][-kwargs.pop("y")]
        for key, value in kwargs.items():
            if hasattr(block, key):
                setattr(block, key, value)

    @property
    def characters(self):
        return self._characters

    @characters.setter
    def characters(self, characters: List[Character]):
        if (
            isinstance(characters, list)
            and len(characters)
            and all([isinstance(c, Character) for c in characters])
        ):
            self._characters = characters

    @property
    def playerID(self):
        return self._playerID

    @playerID.setter
    def playerID(self, playerID: int):
        if isinstance(playerID, int):
            self._playerID = playerID

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color: ColorType):
        if isinstance(color, ColorType):
            self._color = color

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        if score > 0:
            self._score = score

    @property
    def kill(self):
        return self._kill

    @kill.setter
    def kill(self, kill):
        if kill > 0:
            self._kill = kill


if __name__ == "__main__":
    ui = UI(2, ColorType.Black)
    ui.display()
