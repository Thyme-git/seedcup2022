import torch
from resp import *
from req import *
from base import *


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


dir = {0:(lambda x, y : (x-1, y-1)),
           1:(lambda x, y : (x-1,   y)),
           2:(lambda x, y : (  x, y+1)),
           3:(lambda x, y : (x+1, y+1)),
           4:(lambda x, y : (x+1,   y)),
           5:(lambda x, y : (  x, y-1))}

def valid(x, y, obs):
    return x >=0 and y >= 0 and x <= 15 and y <= 15 and obs[1][x][y] == 1

'''return true if no need to attack'''
def check(obs, x, y, d, color = 1):
    next_x, next_y = dir[d](x, y)
    attack_center_x, attack_center_y = dir[d](next_x, next_y)
    if not valid(attack_center_x, attack_center_y, obs):
        return True
    if obs[0][attack_center_x][attack_center_y] != color:
        return False
    for i in range(0, 6):
        nx, ny = dir[d](attack_center_x, attack_center_y)
        if valid(nx, ny, obs) and obs[0][nx][ny] != color:
            return False
    return True

def state2actions(state, playerId = 1):

    obs = state[:-10].view((5, 16, 16))
    index = torch.nonzero(obs[2] == playerId)
    # if state[-10:][1] != 0 and state[-10:][2] != 0:
    #     return -1
    try:
        x, y = index[0][0], index[0][1]
    except:
        return torch.randint(0, 6, (1, )).item()

    d = torch.randint(0, 6, (1, )).item()

    if x == 1 and y <= 7 and y > 1:
        d = 5
    if y == 1 and x >= 1 and x < 11:
        d = 4
    if x == 11 and y >=1 and y < 5:
        d = 2
    if y == 5 and x == 11:
        d = 3
    if x == 12 and y == 6:
        d = 3
    if x == 13 and y == 7:
        d = 2
    if x == 13 and y == 8:
        d = 1
    if x == 12 and y == 8:
        d = 1
    if x == 11 and y >= 8 and y < 11:
        d = 2
    if x == 11 and y == 11:
        d = 3
    if x == 12 and y == 12:
        d = 1
    if x == 11 and y == 12:
        d = 0
    if y == 11 and x > 8 and x <= 10:
        d = 1
    if y == 11 and x == 8:
        d = 2
    if y == 12 and x == 8:
        d = 2
    if y == 13 and x == 8:
        d = 0
    if y == 12 and x == 7:
        d = 0
    if y == 11 and x <= 6 and x > 1:
        d = 1
    if x == 1 and y <= 11 and y > 8:
        d = 5
    if y == 8 and x >= 1 and x < 8:
        d = 4
    if x == 8 and y <= 8 and y > 4:
        d = 5
    if y == 4 and x <= 8 and x > 4:
        d = 1
    if x == 4 and y < 7 and y >= 4:
        d = 2
    if y == 7 and x <= 4 and x > 1:
        d = 1

    if x == 14 and (y == 7 or y == 8):
        d = 1
    if x == 14 and y == 9:
        d = 0
    if x == 15 and (y == 8 or y == 9):
        d = 0
    
    if x == 13 and y == 13:
        d = 0
    if x == 13 and y == 14:
        d = 5
    if x == 14 and y == 13:
        d = 1
    if x == 14 and (y == 14 or y == 15):
        d = 0
    if x == 15 and (y == 14 or y == 15):
        d = 1

    if check(obs, x, y, d):
        d = -d
    return d


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
    no_need_to_attack = False
    if action < 0:
        no_need_to_attack = True
        action = -action
    actions = [ActionReq(characterID,*int2action[action])]
    actions.append(ActionReq(characterID,*int2action[-1]))
    if no_need_to_attack:
        return actions

    if slaveryCD:
        actions.append(ActionReq(characterID,*int2action[8]))
    else:
        actions.append(ActionReq(characterID,*int2action[7]))

    return actions


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