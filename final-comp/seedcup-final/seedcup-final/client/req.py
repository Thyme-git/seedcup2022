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


class ActionParam(JsonBase):
    def __init__(self) -> None:
        super().__init__()


class EmptyActionParam(ActionParam):
    """Placeholder for Action that does not require action param."""

    def __init__(self) -> None:
        super().__init__()


class TurnAroundActionParam(ActionParam):
    """Action param only for TurnAround Action."""

    def __init__(self, turnAroundDirec: Direction) -> None:
        """

        Args:
            turnAroundDirec (Direction): Direction.
        """
        super().__init__()
        self.turnAroundDirec = turnAroundDirec


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


if __name__ == "__main__":
    init_req = InitReq(MasterWeaponType.Durian, SlaveWeaponType.Kiwi)
    init_packet = PacketReq(PacketType.InitReq, [init_req])
    print(init_packet.to_json())

    act_req = ActionReq(1, ActionType.MasterWeaponAttack, EmptyActionParam())
    act_packet = PacketReq(PacketType.ActionReq, [act_req] * 20)
    print(act_packet.to_json())
