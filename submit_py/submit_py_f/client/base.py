import json
from enum import IntEnum, Enum
from typing import Any


class JsonEncoder(json.JSONEncoder):
    """Extended json encoder to support custom class types."""

    def default(self, o: Any) -> Any:
        if issubclass(o.__class__, (JsonIntEnum, JsonBase)):
            return o.to_json()
        return super().default(o)


class JsonIntEnum(IntEnum):
    """Extended int enum to support json Serialization."""

    def to_json(self):
        return self._value_

    def from_json(self, j):
        return self.__class__(int(j))


class JsonBase(object):
    """Base class to support json Serialization and Deserialization.

    Classes that want to support json serialization and deserialization conveniently should
    subclass this class

    Examples:

        class Data(JsonClassBase):
            def __init__(self, dataType: int, data: str) -> None:
                super().__init__()
                self.dataType = dataType
                self.data = data

        >>> data = Data(10, "sss")
        >>> print(data)                        # dump to str, basically similar to json.dumps(data,cls=JsonEncoder)
        >>> {"dataType": 10, "data": {"sss"}}  # <class 'str'>
        >>> print(data.to_json())              # dump to dict
        >>> {'dataType': 10, 'data': {'sss'}}  # <class 'dict'>
        >>> print(data.from_json('{"dataType": 20, "data": "666"}'))
        >>> {"dataType": 20, "data": "666"}
        Hint: Serialization/Deserialization nesting is supported. Suppose data is also a subclass
        of JsonClassBase, serialization will happen the same way.

    """

    def __init__(self) -> None:
        super().__setattr__("_json", {})

    def __setattr__(self, key: str, value: Any) -> None:
        if hasattr(value, "to_json"):
            self._json[key] = value.to_json()
        else:
            self._json[key] = value
        super().__setattr__(key, value)

    def __repr__(self) -> str:
        return json.dumps(self._json, cls=JsonEncoder)

    def __str__(self) -> str:
        return self.__repr__()

    def to_json(self):
        return self._json

    def from_json(self, j: str):
        """Deserialization subclass from str j.

        Be careful! This method will overwrite self.
        **Only support json obj**.

        Args:
            j (str): Str that conforming to the json standard and the serialization type of subclass.

        Returns:
            (subclass): Subclass
        """
        d = json.loads(j)
        if isinstance(d, dict):
            for key, value in d.items():
                if key in self.__dict__:
                    if hasattr(self.__dict__[key], "from_json"):
                        setattr(
                            self, key, self.__dict__[key].from_json(json.dumps(value))
                        )
                    else:
                        setattr(self, key, value)
        return self


class MasterWeaponType(JsonIntEnum):
    PolyWatermelon = 1
    Durian = 2


class SlaveWeaponType(JsonIntEnum):
    Kiwi = 1
    Cactus = 2


class PacketType(JsonIntEnum):
    InitReq = 1
    ActionReq = 2
    ActionResp = 3
    GameOver = 4


class ColorType(JsonIntEnum):
    White = 0
    Red = 1
    Green = 2
    Blue = 3
    Black = 4


class ObjType(JsonIntEnum):
    Null = 0
    Character = 1
    Item = 2
    SlaveWeapon = 3


class Direction(JsonIntEnum):
    Above = 0
    TopRight = 1
    BottomRight = 2
    Bottom = 3
    BottomLeft = 4
    TopLeft = 5


class ResultType(JsonIntEnum):
    Win = 0
    Lose = 1
    Tie = 2


class Emoji(Enum):
    """Kawaii emojis!"""

    WhiteBrick = "⬜"
    RedBrick = "🟥"
    GreenBrick = "🟩"
    BlueBrick = "🟦"
    BlackBrick = "⬛"
    ObstacleBrick = "🧱"
    Character1 = "🦸‍️"
    Character2 = "🦹‍️️"
    Character3 = "🧝️"
    Character4 = "🧛"
    CharacterDead = "💀"
    BuffHp = "❤️"
    BuffSpeed = "⏩"
    SlaveWeaponKiwi = "🥝"
    SlaveWeaponCactus = "🌵"

    def emoji(self):
        return self._value_


if __name__ == "__main__":

    class Data(JsonBase):
        def __init__(self, dataType: int, data: str) -> None:
            super().__init__()
            self.dataType = dataType
            self.data = data

    data = Data(10, "sss")
    print(data)  # dump to str, basically similar to json.dumps(data,cls=JsonEncoder)
    print(data.to_json())  # dump to dict
    print(data.from_json('{"dataType": 20, "data": "666"}'))
