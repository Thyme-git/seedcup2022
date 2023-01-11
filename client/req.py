from base import *
from typing import Union, List


class ActionType(JsonIntEnum):
    """Action space."""

    Move = 1
    TurnAround = 2
    Sneaky = 3
    UnSneaky = 4
    MasterWeaponAttack = 5
    SlaveWeaponAttack = 6


'''action param 基类'''
class ActionParam(JsonBase):
    def __init__(self) -> None:
        super().__init__()



'''placeholder:占位符 for Action that does not require action param'''
class EmptyActionParam(ActionParam):
    """Placeholder for Action that does not require action param."""

    def __init__(self) -> None:
        super().__init__()


'''转向action

class Direction(JsonIntEnum):
    Above = 0
    TopRight = 1
    BottomRight = 2
    Bottom = 3
    BottomLeft = 4
    TopLeft = 5

>>> act = TurnAroundActionParam(你的方向:Direction)
'''
class TurnAroundActionParam(ActionParam):
    """Action param only for TurnAround Action."""

    def __init__(self, turnAroundDirec: Direction) -> None:
        """

        Args:
            turnAroundDirec (Direction): Direction.
        """
        super().__init__()
        self.turnAroundDirec = turnAroundDirec


'''
初始化选择主武器以及副武器
class MasterWeaponType(JsonIntEnum):
    PolyWatermelon = 1
    Durian = 2

class SlaveWeaponType(JsonIntEnum):
    Kiwi = 1           # 猕猴桃
    Cactus = 2         # 仙人掌

>>> init_req = InitReq(主武器， 副武器)
'''
class InitReq(JsonBase):
    """Init request payload."""

    def __init__(
        self,
        masterWeaponType: MasterWeaponType,
        slaveWeaponType: SlaveWeaponType,
    ) -> None:
        super().__init__()
        self.masterWeaponType = masterWeaponType
        self.slaveWeaponType = slaveWeaponType


class ActionReq(JsonBase):
    """Action request payload."""

    def __init__(
        self, characterID: int, actionType: ActionType, actionParam: ActionParam
    ) -> None:
        super().__init__()
        self.characterID = characterID
        self.actionType = actionType
        self.actionParam = actionParam


class PacketReq(JsonBase):
    """The basic packet of communication with the server."""

    def __init__(
        self, type: PacketType, data: Union[List[InitReq], List[ActionReq]]
    ) -> None:
        super().__init__()
        self.type = type
        self.data = data


'''for test'''
if __name__ == "__main__":
    init_req = InitReq(MasterWeaponType.Durian, SlaveWeaponType.Kiwi)
    init_packet = PacketReq(PacketType.InitReq, [init_req])
    print(init_packet.to_json())

    act_req = ActionReq(1, ActionType.MasterWeaponAttack, EmptyActionParam())
    act_packet = PacketReq(PacketType.ActionReq, [act_req] * 20)
    print(act_packet.to_json())
