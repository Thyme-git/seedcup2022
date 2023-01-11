import torch.nn as nn
import torch
import torch.nn.functional as F
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


'''
   characterState: {
        "x": 5, "y": -10, 
        "playerID": 1, 
        "characterID": 0, 
        "direction": 3, 
        "color": 2, 
        "hp": 50, 
        "moveCD": 4, 
        "moveCDLeft": 0, 
        "isAlive": true, 
        "isSneaky": false, 
        "isGod": false, 
        "rebornTimeLeft": 0, 
        "godTimeLeft": 0, 

        "slaveWeapon": {"weaponType": 1, "attackCD": 30, "attackCDLeft": 0}, 
        "masterWeapon": {"weaponType": 1, "attackCD": 3, "attackCDLeft": 0}}

'''

'''参考main.py里的refreshUI()'''
def actionResp2info(packet: PacketResp):
    '''将actionResp转换成reward, state, done, slavryCD'''
    if packet.type == PacketType.GameOver:
        return packet.data.scores[-1], torch.zeros((5*16*16+10,)), True, False

    data = packet.data
    score = data.score
    playerId = data.playerID
    kill = data.kill
    color = data.color
    blocks =  data.map.blocks

    # 更加专注于拿下格子
    # total_reward = score - kill*9
    total_reward = score
    done = False

    # [0]:颜色信息， [1] 墙体信息， [2]:人物, [3]:buff, [4]:slaveryweapon
    state = torch.zeros((5, 16, 16))
    player_state = torch.zeros((10,))

    slavryCD = False

    for block in blocks:
        state[0][block.x][-block.y] = 1.0 if block.color == color else -1.0
        state[1][block.x][-block.y] = 1.0 if block.valid else 0.0
        if not block.valid:
            state[0][block.x][-block.y] = 0.0
        if len(block.objs):
            for obj in block.objs:
                if obj.type == ObjType.Character:
                    if obj.status.playerID == playerId:
                        obj.status.moveCD

                        player_state[0] = obj.status.hp
                        player_state[1] = obj.status.masterWeapon.attackCDLeft
                        player_state[2] = obj.status.slaveWeapon.attackCDLeft
                        player_state[3] = obj.status.rebornTimeLeft
                        player_state[4] = obj.status.godTimeLeft
                        if not obj.status.isAlive:
                            total_reward -= 1
                        if obj.status.slaveWeapon.attackCDLeft == 0:
                            slavryCD = True
                        state[2][block.x][-block.y] = 1
                    else:
                        player_state[5] = obj.status.hp
                        player_state[6] = obj.status.moveCD
                        player_state[7] = obj.status.moveCDLeft
                        player_state[8] = obj.status.rebornTimeLeft
                        player_state[9] = obj.status.godTimeLeft
                        state[2][block.x][-block.y] = -1
    
                elif obj.type == ObjType.Item:
                    state[3][block.x][-block.y] += obj.status.buffType + 1 
   
                elif obj.type == ObjType.SlaveWeapon:
                    if obj.status.playerID == playerId:
                        state[4][block.x][-block.y] = 1
                    else:
                        state[4][block.x][-block.y] = -1

    return total_reward, \
           torch.concat([state.view(-1), player_state]), \
           done, \
           slavryCD


def state2actions(state, playerId = 1):
    global count
    obs = state[:-10].view((5, 16, 16))
    index = torch.nonzero(obs[2] == playerId)
    # if state[-10:][1] != 0 and state[-10:][2] != 0:
    #     return -1
    try:
        x, y = index[0][0], index[0][1]
    except:
        return torch.randint(0, 6, (1, )).item()
    
    # if x == 1 and y <= 11 and y > 1:
    if x == 1 and y <= 7 and y > 1:
        return 5
    if y == 1 and x >= 1 and x < 11:
        return 4
    if x == 11 and y >=1 and y < 11:
        return 2
    if y == 11 and x > 1 and x <= 11:
        return 1
    if x == 1 and y <= 11 and y > 8:
        return 5
    if y == 8 and x >= 1 and x < 8:
        return 4
    if x == 8 and y <= 8 and y > 4:
        return 5
    if y == 4 and x <= 8 and x > 4:
        return 1
    if x == 4 and y < 7 and y >= 4:
        return 2
    if y == 7 and x <= 4 and x > 1:
        return 1
    
    return torch.randint(0, 6, (1, )).item()


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
def action2actionReq(action, characterID, slaveryCD):
    if action  == -1:
        return []
    actions = [ActionReq(characterID,*int2action[action])]
    actions.append(ActionReq(characterID,*int2action[-1]))
    if slaveryCD:
        actions.append(ActionReq(characterID,*int2action[8]))
    else:
        actions.append(ActionReq(characterID,*int2action[7]))

    return actions


def refreshUI(ui: UI, packet: PacketResp):
    """Refresh the UI according to the response."""
    data = packet.data
    if packet.type == PacketType.ActionResp:
        ui.playerID = data.playerID
        ui.color = data.color
        ui.characters = data.characters
        ui.score = data.score
        ui.kill = data.kill

        for block in data.map.blocks:
            if len(block.objs):
                ui.block = {
                    "x": block.x,
                    "y": block.y,
                    "color": block.color,
                    "valid": block.valid,
                    "obj": block.objs[-1].type,
                    "data": block.objs[-1].status,
                }
            else:
                ui.block = {
                    "x": block.x,
                    "y": block.y,
                    "color": block.color,
                    "valid": block.valid,
                    "obj": ObjType.Null,
                }
    subprocess.run(["clear"])
    ui.display()


def main():
    '''🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡'''
    with Client() as client:
        client.connect()
        init_req = InitReq(MasterWeaponType.Durian, SlaveWeaponType.Cactus)
        init_packet = PacketReq(PacketType.InitReq, init_req)
        client.send(init_packet)
        resp = client.recv()
        
        total_reward, state, done, slaveryCD = actionResp2info(resp)
        characterId = resp.data.characters[-1].characterID
        
        while resp.type != PacketType.GameOver:
            action_int = state2actions(state)
            actionReq = action2actionReq(action_int, characterId, slaveryCD)
            actionPacket = PacketReq(PacketType.ActionReq, actionReq)
            client.send(actionPacket)
            resp = client.recv()
            
            next_total_reward, next_state, done, slaveryCD = actionResp2info(resp)
            state = next_state

if __name__ == '__main__':
    from main import Client
    main()