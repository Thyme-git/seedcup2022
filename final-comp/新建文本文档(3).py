import torch
from resp import *
from req import *
from base import *
from collections import deque
import random

import os
import json
from threading import Thread

from ui import UI
import subprocess

from time import sleep
from functools import partial


def actionResp2info(packet: PacketResp):
    '''将actionResp转换成reward, state, done, slavryCD'''
    if packet.type == PacketType.GameOver:
        return packet.data.scores[-1], torch.zeros((5 * 16 * 16 + 10,)), True, False

    data = packet.data
    playerId = data.playerID
    color = data.color
    blocks = data.map.blocks
    done = False

    '''1:标记无视野'''
    # [0]:颜色信息， [1] 墙体信息， [2]:人物, [3]:buff, [4]:slavery_weapon    -10标记无视野
    state = torch.zeros((5, 24, 24))
    player_state = torch.zeros((10,))

    slaveryCD = False

    '''2:有可能看不到敌人'''
    CON = []
    PRO = []

    for block in blocks:
        state[0][block.x][-block.y] = 1.0 if block.color == color else -1.0
        state[1][block.x][-block.y] = 1.0 if block.valid else 0.0
        if len(block.objs):
            for obj in block.objs:
                if obj.type == ObjType.Character:
                    if obj.status.playerID == playerId:
                        player_state[0] = obj.status.hp
                        player_state[1] = obj.status.masterWeapon.attackCDLeft
                        player_state[2] = obj.status.slaveWeapon.attackCDLeft
                        player_state[3] = obj.status.rebornTimeLeft
                        player_state[4] = obj.status.godTimeLeft
                        if obj.status.slaveWeapon.attackCDLeft == 0:
                            slaveryCD = True
                        state[2][block.x][-block.y] = 1
                        if obj.status.isAlive:
                            PRO.append(obj.status)
                    else:
                        player_state[5] = obj.status.hp
                        player_state[6] = obj.status.moveCD
                        player_state[7] = obj.status.moveCDLeft
                        player_state[8] = obj.status.rebornTimeLeft
                        player_state[9] = obj.status.godTimeLeft
                        state[2][block.x][-block.y] = -1
                        if obj.status.isAlive:
                            CON.append(obj.status)

                elif obj.type == ObjType.Item:
                    state[3][block.x][-block.y] += obj.status.buffType + 1

                elif obj.type == ObjType.SlaveWeapon:
                    if obj.status.playerID == playerId:
                        state[4][block.x][-block.y] = 1
                    else:
                        state[4][block.x][-block.y] = -1

        # elif len(block.objs) == 0 and block.color != color:
        #     state[0][block.x][-block.y] = -1.0
        # '''3:无视野判断'''
        if not block.valid:
            state[0][block.x][-block.y] = 0.0

    '''两个角色防止混淆'''
    if len(PRO) > 1:
        if PRO[0].characterID > PRO[1].characterID:
            PRO[0], PRO[1] = PRO[1], PRO[0]

    return torch.concat([state.view(-1), player_state]), slaveryCD, (CON, PRO)


count = 0
count1 = 0
count2 = 0
count3 = 0


def Get_rand(pos = (7, 7)):
    print('random!!!')
    if pos[0] == 0:
        return torch.randint(3, 5, (1,)).item()
    if pos[1] == 0:
        return torch.randint(2, 4, (1,)).item()
    return torch.randint(0, 6, (1,)).item()


def Distance(x, y):
    return torch.dist(x + 0.0, y + 0.0, p = 2)
    # return torch.dist(x + 0.0, y + 0.0, p = 1)


def Select_dir(pos, aim_pos):
    unit_dir = torch.tensor([[-1, -1], [-1, 0], [0, 1], [1, 1], [1, 0], [0, -1]])
    distances = torch.tensor(list(map(partial(Distance, aim_pos), pos + unit_dir)))
    optimal_dir = torch.argmin(distances)
    return optimal_dir.item()


def Nearest(pos, aims):
    distances = torch.tensor(list(map(partial(Distance, pos), aims)))
    nearest = torch.argmin(distances)
    return aims[nearest]


def Chase_filter(pos, aims, go_Vacant: bool = False):
    filter1 = aims[[i for i in range(aims.shape[0]) if Distance(aims[i], pos) > int(go_Vacant)]]
    filter2 = filter1[[i for i in range(filter1.shape[0]) if aims[i, 0] < 21 and aims[i, 1] < 21]]
    return filter2


def state2actions(state, characters):
    global count, count1, count2, count3

    obs = state[:-10].view((5, 24, 24))
    CON, PRO = characters[0], characters[1]
    pos = [None, None]
    if len(PRO) >= 1:
        pos[0] = torch.tensor([PRO[0].x, -PRO[0].y])
    if len(PRO) == 2:
        pos[1] = torch.tensor([PRO[1].x, -PRO[1].y])
    
    aim_pos = [None, None]
    if len(CON) >= 1:
        aim_pos[0] = torch.tensor([CON[0].x, -CON[0].y])
    if len(CON) == 2:
        aim_pos[1] = torch.tensor([CON[1].x, -CON[1].y])


    if len(PRO) + len(CON) >= 3:
        if len(PRO) == 1:
            if Distance(pos[0], aim_pos[0]) > Distance(pos[0], aim_pos[1]):
                tmp = aim_pos[0]
                aim_pos[0] = aim_pos[1]
                aim_pos[1] = tmp
        else:
            if Distance(pos[0], aim_pos[0]) > Distance(pos[1], aim_pos[0]):
                tmp = pos[0]
                pos[0] = pos[1]
                pos[1] = tmp

    # if count == 0 and pos[0][0] == 21 and pos[0][1] == 11:
    #     count = 1

    # if not count:
    #    return certainRoute1(PRO), certainRoute2(PRO)

    d = [-1, -1]
    for i in range(len(PRO)):
        if aim_pos[i] is not None:
            d[i] = Select_dir(pos[i], aim_pos[i])
            # if Distance(aim_pos[i], pos[i]) < 2:
            #     d[i] = (d[i] + 3) % 6
            continue
        vacant_indices = Chase_filter(pos[i], torch.nonzero(obs[0] == -1), True)
        speed_indices = Chase_filter(pos[i], torch.nonzero(obs[3] == 2))
        hp_indices = Chase_filter(pos[i], torch.nonzero(obs[3] == 3))
        if PRO[i].hp <= 50:
            if hp_indices.shape[0] != 0:
                hp_pos = Nearest(pos[i], hp_indices)
                d[i] = Select_dir(pos[i], hp_pos)
            elif vacant_indices.shape[0] != 0 and d[i] == -1:
                vacant_pos = Nearest(pos[i], vacant_indices)
                d[i] = Select_dir(pos[i], vacant_pos)
        else:
            if speed_indices.shape[0] != 0 and PRO[i].moveCD >= 2:
                speed_pos = Nearest(pos[i], speed_indices)
                d[i] = Select_dir(pos[i], speed_pos)
            if hp_indices.shape[0] != 0 and PRO[i].hp < 100 and d[i] == -1:
                hp_pos = Nearest(pos[i], hp_indices)
                d[i] = Select_dir(pos[i], hp_pos)
            if vacant_indices.shape[0] != 0 and d[i] == -1:
                vacant_pos = Nearest(pos[i], vacant_indices)
                d[i] = Select_dir(pos[i], vacant_pos)
        if len(PRO) == 2 and Distance(pos[0], pos[1]) < 2:
            d[1] = (Select_dir(pos[1], pos[0]) + 3) % 6

    # if len(CON) == 1 and CON[0].isAlive and not CON[0].isGod:
    #     if len(PRO) == 2:
    #         i = 0 if Distance(pos[0], aim_pos[0]) < Distance(pos[1], aim_pos[0]) else 1
    #     elif len(PRO) == 1:
    #         i = 0
    #     if len(PRO) > 0:
    #         d[i] = Select_dir(pos[i], aim_pos[0])
    # if len(CON) == 2:
    #     ii = -1
    #     for i in range(2):
    #         if CON[i].isAlive and not CON[i].isGod:
    #             if ii == -1:
    #                 if len(PRO) == 2:
    #                     ii = 0 if Distance(pos[0], aim_pos[i]) < Distance(pos[1], aim_pos[i]) else 1
    #                 elif len(PRO) == 1:
    #                     ii = 0
    #                 if len(PRO) > 0:
    #                     d[ii] = Select_dir(pos[ii], aim_pos[0])
    #             elif len(PRO) == 2:
    #                 ii = 0 if ii == 1 else 1
    #                 d[ii] = Select_dir(pos[ii], aim_pos[0])

    return tuple(d)


int2action = {
    -1: (ActionType.Move, EmptyActionParam()),
    0: (ActionType.TurnAround, TurnAroundActionParam(Direction.Above)),
    1: (ActionType.TurnAround, TurnAroundActionParam(Direction.TopRight)),
    2: (ActionType.TurnAround, TurnAroundActionParam(Direction.BottomRight)),
    3: (ActionType.TurnAround, TurnAroundActionParam(Direction.Bottom)),
    4: (ActionType.TurnAround, TurnAroundActionParam(Direction.BottomLeft)),
    5: (ActionType.TurnAround, TurnAroundActionParam(Direction.TopLeft)),
    7: (ActionType.MasterWeaponAttack, EmptyActionParam()),
    8: (ActionType.SlaveWeaponAttack, EmptyActionParam()),
}


def state2actionReq(state, characters):
    direction = state2actions(state, characters)
    PRO = characters[1]

    actions = []

    if len(PRO) >= 1 and direction[0] != -1:
        actions = [ActionReq(PRO[0].characterID, *int2action[direction[0]])]
        actions.append(ActionReq(PRO[0].characterID, *int2action[8 if PRO[0].slaveWeapon.attackCDLeft == 0 else 7]))
        actions.append(ActionReq(PRO[0].characterID, *int2action[-1]))
    if len(PRO) == 2 and direction[1] != -1:
        actions.append(ActionReq(PRO[1].characterID, *int2action[direction[1]]))
        actions.append(ActionReq(PRO[1].characterID, *int2action[8 if PRO[1].slaveWeapon.attackCDLeft == 0 else 7]))
        actions.append(ActionReq(PRO[1].characterID, *int2action[-1]))
    return actions


# def routeTest(PRO):
#     # CON, PRO = characters[0], characters[1]
#     dir = certainRoute(PRO)

#     actions = [ActionReq(PRO[0].characterID, *int2action[dir[0]])]
#     actions.append(ActionReq(PRO[0].characterID, *int2action[8 if PRO[0].slaveWeapon.attackCDLeft == 0 else 7]))
#     actions.append(ActionReq(PRO[0].characterID, *int2action[-1]))

#     actions.append(ActionReq(PRO[1].characterID, *int2action[dir[1]]))
#     actions.append(ActionReq(PRO[1].characterID, *int2action[8 if PRO[1].slaveWeapon.attackCDLeft == 0 else 7]))
#     actions.append(ActionReq(PRO[1].characterID, *int2action[-1]))
#     return actions


def refreshUI(ui: UI, packet: PacketResp):
    """Refresh the UI according to the response."""
    data = packet.data
    if packet.type == PacketType.ActionResp:
        ui.playerID = data.playerID
        ui.color = data.color
        ui.characters = data.characters
        ui.score = data.score
        ui.kill = data.kill
        ui.frame = data.frame

        for block in data.map.blocks:
            if len(block.objs):
                ui.block = {
                    "x": block.x,
                    "y": block.y,
                    "color": block.color,
                    "valid": block.valid,
                    "frame": block.frame,
                    "obj": block.objs[-1].type,
                    "data": block.objs[-1].status,
                }
            else:
                ui.block = {
                    "x": block.x,
                    "y": block.y,
                    "color": block.color,
                    "valid": block.valid,
                    "frame": block.frame,
                    "obj": ObjType.Null,
                }
    subprocess.run(["clear"])
    ui.display()


def main():
    ui = UI()

    '''🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡'''
    with Client() as client:
        client.connect()
        init_req = InitReq(MasterWeaponType.Durian, SlaveWeaponType.Cactus)
        # init_req1 = InitReq(MasterWeaponType.PolyWatermelon, SlaveWeaponType.Kiwi)
        init_packet = PacketReq(PacketType.InitReq, [init_req] * 2)
        client.send(init_packet)
        resp = client.recv()
        # print(resp.data.characters)

        refreshUI(ui, resp)

        '''3:我方的characterId'''
        characterIds = [resp.data.characters[0].characterID, resp.data.characters[0].characterID]

        while resp.type != PacketType.GameOver:
            next_state, slaveryCD, characters = actionResp2info(resp)
            state = next_state

            '''4:need to be modify'''
            actionReq = state2actionReq(state, characters)

            actionPacket = PacketReq(PacketType.ActionReq, actionReq)
            client.send(actionPacket)
            resp = client.recv()

            refreshUI(ui, resp)


if __name__ == '__main__':
    from main import Client

    main()
