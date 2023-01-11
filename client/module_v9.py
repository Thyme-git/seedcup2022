import torch.nn as nn
import torch
import torch.nn.functional as F
from resp import *
from req import *
from base import *
from collections import deque
import random
import numpy as np

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
        if len(block.objs):
            for obj in block.objs:
                if obj.type == ObjType.Character:
                    if obj.status.playerID == playerId:
                        player_state[0] = obj.status.hp
                        player_state[1] = obj.status.moveCD
                        player_state[2] = obj.status.moveCDLeft
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


'''
    action space : [0, 5]
        0 --> 5 : turn around 0-5 and move and use master/slavery weapon
'''
# class Qnet(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         self.fc1 = nn.Linear(5*16*16+10, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc_A = nn.Linear(256, 6)
#         self.fc_V = nn.Linear(256, 1)
    
#     def forward(self, x):
#         x = x.view((x.size()[0], -1))
#         A = self.fc_A(F.relu(self.fc2(F.relu(self.fc1(x)))))
#         V = self.fc_V(F.relu(self.fc2(F.relu(self.fc1(x)))))
#         Q = V + A - A.mean(1).view(-1, 1)
#         return Q

class PolicyNet(torch.nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(5*16*16+10, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 6)

    def forward(self, x):
        x = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        return F.softmax(x, dim=1)


class QValueNet(torch.nn.Module):
    def __init__(self):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(5*16*16+10, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

class Config:
    alpha_lr = 1e-3
    target_entropy = -1
    sigma = 0.01
    action_dim = 6
    tau = 0.005  # 软更新参数
    epochs = 10
    lmbda = 0.95
    n_epoch = 6000
    action_space = 6
    actor_lr = 1e-3
    critic_lr = 1e-3
    gamma = 0.98
    epsilon = 0.2
    target_update = 100
    buffer_size = 500
    minimal_size = 100
    batch_size = 32
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

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

         # 策略网络
        self.actor = PolicyNet().to(config.device)
        # 第一个Q网络
        self.critic_1 = QValueNet().to(config.device)
        # 第二个Q网络
        self.critic_2 = QValueNet().to(config.device)
        self.target_critic_1 = QValueNet().to(config.device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet().to(config.device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=config.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=config.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=config.critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=config.alpha_lr)
        self.target_entropy = config.target_entropy  # 目标熵的大小
        self.gamma = config.gamma
        self.tau = config.tau
        self.device = config.device
        self.epsilon = config.epsilon

        self.memory = Memory(config.buffer_size)

    def act(self, state):
        if (torch.randn((1,)).item() < self.epsilon):
            self.epsilon *= self.config.gamma
            action = random.randint(0, self.config.action_space-1)
            return action

        probs = self.actor(state.to(self.device).view((1, *state.size())))
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +param.data * self.tau)

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

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states).gather(1, actions.view((-1, 1)))
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions.view((-1, 1)))
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def save(self, path_actor = './model_v9_actor.pt', path_critic_1 = './model_v9_critic_1.pt', path_critic_2 = './model_v9_critic_2.pt'):
        torch.save(self.actor.state_dict(), path_actor)
        torch.save(self.critic_1.state_dict(), path_critic_1)
        torch.save(self.critic_2.state_dict(), path_critic_2)

    def load(self, path_actor = './model_v9_actor.pt', path_critic_1 = './model_v9_critic_1.pt', path_critic_2 = './model_v9_critic_2.pt'):
        self.actor.load_state_dict(torch.load(path_actor))
        self.critic_1.load_state_dict(torch.load(path_critic_1))
        self.critic_2.load_state_dict(torch.load(path_critic_2))
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())


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
    obs = state[:-10].view((5, 16, 16))
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
    ports = [9994, 9995, 9996, 9997, 9998, 9999]
    # ports = [9997, 9998, 9999]
    if UserInterface:
        ui = UI()
    
    config = Config()
    agent = Agent(config)
    agent.load()

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

                init_req = InitReq(MasterWeaponType.Durian, SlaveWeaponType.Cactus)
                init_packet = PacketReq(PacketType.InitReq, init_req)
                client.send(init_packet)


                resp = client.recv()
                if UserInterface:
                    refreshUI(ui, resp)
                
                total_reward, state, done, slaveryCD = actionResp2info(resp)

                characterId = resp.data.characters[-1].characterID

                while resp.type != PacketType.GameOver:
                    action_int = agent.act(state)
                    actionReq = action2actionReq(action_int, characterId, slaveryCD)
                    actionPacket = PacketReq(PacketType.ActionReq, actionReq)
                    client.send(actionPacket)

                    resp = client.recv()
                    if UserInterface:
                        refreshUI(ui, resp)
                    
                    next_total_reward, next_state, done, slaveryCD = actionResp2info(resp)

                    reward = next_total_reward-total_reward
                    check = getNextLoc(action_int, state)
                    if check == 0:
                        reward += 30
                    elif check == 1:
                        reward -= 20
                    elif check == 2:
                        reward -= 20
                    elif check == 3:
                        reward -= 20
                    
                    agent.memorize(state, action_int,reward, next_state, done)
                    agent.learn()

                    state = next_state
                    total_reward = next_total_reward

                agent.save()
                scores = resp.data.scores
                with open('./score_logs_v9', 'a') as f:
                    f.write(str(scores) + '\n')
                    f.close()


        serverThead.join()
        botThead.join()
        sleep(2)


if __name__ == '__main__':
    from main import Client
    main()