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
    '''将actionResp转换成reward, state, done'''
    if packet.type == PacketType.GameOver:
        return packet.data.scores[-1], torch.zeros((5*16*16+12,)), True

    data = packet.data
    score = data.score
    playerId = data.playerID
    kill = data.kill
    color = data.color
    blocks =  data.map.blocks
    total_reward = score
    done = False

    # [0]:颜色信息， [1] 墙体信息， [2]:人物, [3]:buff, [4]:slaveryweapon
    state = torch.zeros((5, 16, 16))
    player_state = torch.zeros((12,))

    for block in blocks:
        state[0][block.x][-block.y] = 1.0 if block.color == color else -1.0
        state[1][block.x][-block.y] = 1.0 if block.valid else 0.0
        if len(block.objs):
            for obj in block.objs:
                if obj.type == ObjType.Character:
                    if obj.status.playerID == playerId:
                        player_state[0] = obj.status.hp
                        player_state[1] = obj.status.moveCD
                        player_state[2] = obj.status.moveCDLeft
                        player_state[3] = obj.status.rebornTimeLeft
                        player_state[4] = obj.status.godTimeLeft
                        player_state[5] = obj.status.slaveWeapon.attackCDLeft
                        player_state[6] = obj.status.masterWeapon.attackCDLeft
                        if not obj.status.isAlive:
                            total_reward -= 1
                        state[2][block.x][-block.y] = 1
                    else:
                        player_state[7] = obj.status.hp
                        player_state[8] = obj.status.moveCD
                        player_state[9] = obj.status.moveCDLeft
                        player_state[10] = obj.status.rebornTimeLeft
                        player_state[11] = obj.status.godTimeLeft
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
           done


'''
    action space : [0, 5]
        0 --> 5 : turn around 0-5 and move and use master/slavery weapon
'''
class Qnet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(5*16*16+12, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_A = nn.Linear(256, 91)
        self.fc_V = nn.Linear(256, 1)
    
    def forward(self, x):
        x = x.view((x.size()[0], -1))
        A = self.fc_A(F.relu(self.fc2(F.relu(self.fc1(x)))))
        V = self.fc_V(F.relu(self.fc2(F.relu(self.fc1(x)))))
        Q = V + A - A.mean(1).view(-1, 1)
        return Q
    

class Config:
    n_epoch = 6000
    action_space = 91
    lr = 1e-3
    gamma = 0.99
    epsilon = 0.2
    target_update = 50
    buffer_size = 50000
    minimal_size = 10000
    batch_size = 256
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")


class Memory:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def push(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))
    
    def size(self):
        return len(self.buffer)
    
    def sample(self, batch_size):
        transition = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transition)
        return torch.stack(states, 0),\
               torch.tensor(actions),\
               torch.tensor(rewards), \
               torch.stack(next_states, 0), \
               torch.tensor(dones).to(torch.float)


class Agent:
    def __init__(self, config:Config):
        self.config = config

        self.epsilon = config.epsilon
    
        self.memory = Memory(config.buffer_size)

        self.q_net = Qnet().to(self.config.device)
        self.target_q_net = Qnet().to(self.config.device)
        
        self.criterion = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr=config.lr)

        self.count = 0

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=6000)

    def act(self, state):
        if (torch.randn((1,)).item() < self.epsilon):
            self.epsilon *= self.config.gamma
            action = random.randint(0, self.config.action_space-1)
        else:
            action = self.q_net(state.view((1, *state.size())).to(self.config.device)).argmax(1).item()
        return action

    def memorize(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if (self.memory.size() < self.config.minimal_size):
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        states = states.to(self.config.device)
        actions = actions.to(self.config.device)
        rewards = rewards.to(self.config.device)
        next_states = next_states.to(self.config.device)
        dones = dones.to(self.config.device)

        q_values = self.q_net(states).gather(1, actions.view((-1, 1)))

        max_actions = self.q_net(next_states).argmax(1)

        next_q_values = self.target_q_net(next_states).gather(1, max_actions.view((-1, 1)))

        q_targets = rewards.view((-1, 1)) + self.config.gamma*next_q_values*(1-dones.view((-1, 1)))

        loss = self.criterion(q_values.view(-1), q_targets.view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        
        if self.count % self.config.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.count += 1

    def save(self, path = './model.pt'):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path = './model.pt'):
        self.q_net.load_state_dict(torch.load(path))
        self.target_q_net.load_state_dict(torch.load(path))


int2dir = [
    (ActionType.TurnAround, TurnAroundActionParam(Direction.Above)),
    (ActionType.TurnAround, TurnAroundActionParam(Direction.TopRight)),
    (ActionType.TurnAround, TurnAroundActionParam(Direction.BottomRight)),
    (ActionType.TurnAround, TurnAroundActionParam(Direction.Bottom)),
    (ActionType.TurnAround, TurnAroundActionParam(Direction.BottomLeft)),
    (ActionType.TurnAround, TurnAroundActionParam(Direction.TopLeft)),
]
int2weapon = [
    (ActionType.MasterWeaponAttack, EmptyActionParam()),
    (ActionType.SlaveWeaponAttack, EmptyActionParam()),
]
move = (ActionType.Move, EmptyActionParam())
def int2actionReq(action:int, characterID):
    i = action%7
    j = action//7
    actions = []
    if i != 6:
        actionReq = ActionReq(characterID, *int2dir[i])
        actions.append(actionReq)
        actionReq = ActionReq(characterID, *move)
        actions.append(actionReq)
    if j != 12:
        actionReq = ActionReq(characterID, *int2dir[j%6])
        actions.append(actionReq)
        actionReq = ActionReq(characterID, *int2weapon[j//6])
        actions.append(actionReq)

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


'''
    坐标(x, y):

    方向s:      到达:
        0   --> (x-1, y-1)
        1   --> (x-1,   y)
        2   --> (  x, y+1)
        3   --> (x+1, y+1)
        4   --> (x+1,   y)
        5   --> (  x, y-1)
'''
dir = {0:(lambda x, y : (x-1, y-1)),
           1:(lambda x, y : (x-1,   y)),
           2:(lambda x, y : (  x, y+1)),
           3:(lambda x, y : (x+1, y+1)),
           4:(lambda x, y : (x+1,   y)),
           5:(lambda x, y : (  x, y-1))}
grayLocations = [(13, 7), (13, 8), (14, 7), (14, 8), (14, 9), (15, 8), (15, 9), (13,13), (13, 14), (14, 13), (14, 14), (14, 15), (15, 14), (15, 15)]
# 撞墙扣除CD
'''吃到了buff:0, 呆在自己的领地上:1, 进入灰色地带:2, 撞墙:3, 其他:4'''
def getNextLoc(action_int, state, playId=1, color=1):
    action_int = action_int%7
    if action_int == 6:
        return 4
    obs = state[:-12].view((5, 16, 16))
    index = torch.nonzero(obs[2] == playId)
    if index.size()[0] == 0:
        return 4
    x, y = index[0][0], index[0][1]
    
    next_x, next_y = dir[action_int](x, y)
    if next_x < 0 or next_y < 0 or next_x > 15 or next_y > 15 or obs[1][next_x][next_y] == 0:
        return 3
    if (next_x, next_y) in grayLocations:
        return 2
    if obs[0][next_x][next_y] == color:
        return 1
    if obs[3][next_x][next_y] > 0:
        return 0
    return 4


# 修改config.json来改变端口实现无缝衔接训练
def changePort(port:int = 9999):
    with open('./config.json') as f:
        config_json = json.load(f)
        config_json['Port'] = port
        f.close()
    with open('./config.json', 'w') as f:
        json.dump(config_json, f)
        f.close()


UserInterface = 1
def main():
    '''🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡🥵🤡'''
    ports = [9991, 9992, 9993, 9994, 9995, 9996, 9997, 9998, 9999]
    # ports = [9998, 9999]
    if UserInterface:
        ui = UI()
    
    config = Config()
    agent = Agent(config)
    # agent.load('./model_v7.pt')

    for epoch in range(config.n_epoch):
        port = ports[epoch%len(ports)]
        changePort(port)
        serverThead = Thread(target=os.system, args=('./seedcupServer',))
        botThead = Thread(target=os.system, args=('./bot',))
        
        serverThead.start()
        sleep(2)
        botThead.start()
        sleep(2)

        scores = 0

        connected = False

        with Client() as client:
                client.port = port
                client.connect()

                init_req = InitReq(MasterWeaponType.PolyWatermelon, SlaveWeaponType.Cactus)
                init_packet = PacketReq(PacketType.InitReq, init_req)
                client.send(init_packet)


                resp = client.recv()
                if UserInterface:
                    refreshUI(ui, resp)
                
                total_reward, state, done = actionResp2info(resp)

                # print(state[0])
                # print(state[1])
                # print(state[2])
                # print(state[3])
                # print(state[4])

                characterId = resp.data.characters[-1].characterID

                while resp.type != PacketType.GameOver:
                    action_int = agent.act(state)
                    actionReq = int2actionReq(action_int, characterId)
                    actionPacket = PacketReq(PacketType.ActionReq, actionReq)
                    client.send(actionPacket)

                    resp = client.recv()
                    if UserInterface:
                        refreshUI(ui, resp)
                    
                    next_total_reward, next_state, done = actionResp2info(resp)

                    reward = next_total_reward-total_reward
                    
                    check = getNextLoc(action_int, state)
                    if check == 0:
                        reward += 10
                    elif check == 1:
                        reward -= 2
                    elif check == 2:
                        reward -= 5
                    elif check == 3:
                        reward -= 5
                    
                    agent.memorize(state, action_int,reward, next_state, done)
                    agent.learn()

                    state = next_state
                    total_reward = next_total_reward

                agent.save('./model_v7.pt')
                scores = resp.data.scores
                with open('./score_logs_v7', 'a') as f:
                    f.write(str(scores) + '\n')
                    f.close()


        serverThead.join()
        botThead.join()
        sleep(2)


if __name__ == '__main__':
    from main import Client
    main()