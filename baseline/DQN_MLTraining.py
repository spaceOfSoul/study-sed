import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import subprocess
import time
import signal
import sys
from models import DQN

# Hyperparameters
num_parameters = 14
num_actions = num_parameters * 6  #prameter change rnage (1~6)
epsilon = 0.1  # 탐색 vs. 활용 비율
alpha = 0.001  # 학습률
gamma = 0.9    # 할인율
batch_size = 32
memory_size = 2000
target_update = 10  # 타깃 네트워크 업데이트 주기

# 경험 재생 메모리
memory = deque(maxlen=memory_size)


# 모델 초기화
policy_net = DQN(num_parameters, num_actions)
target_net = DQN(num_parameters, num_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
criterion = nn.MSELoss()

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)  # 탐색
    else:
        with torch.no_grad():
            return policy_net(state).argmax().item()  # 활용

def optimize_model():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    state_batch = torch.cat([s for (s, _, _, _) in batch])
    action_batch = torch.tensor([a for (_, a, _, _) in batch]).unsqueeze(1)
    reward_batch = torch.tensor([r for (_, _, r, _) in batch])
    next_state_batch = torch.cat([s for (_, _, _, s) in batch])

    current_q_values = policy_net(state_batch).gather(1, action_batch)
    next_q_values = target_net(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + (gamma * next_q_values)

    loss = criterion(current_q_values, expected_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def run_training_process(parameter_file, log_file):
    return subprocess.Popen(
        ['python', 'main.py', '-dir', parameter_file, '|', 'tee', log_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setpgrp,
        shell=False
    )

def evaluate_model(state):
    # 파라미터 파일 생성
    parameter_file = 'parameters.txt'
    log_file = 'training.log'
    with open(parameter_file, 'w') as f:
        f.write(','.join(map(str, state.squeeze().numpy())))

    # 학습 프로세스 실행
    proc = run_training_process(parameter_file, log_file)

    try:
        # 학습 프로세스가 완료될 때까지 대기
        while proc.poll() is None:
            time.sleep(1)

    except KeyboardInterrupt:
        # DQN 프로세스가 강제로 종료된 경우, 자식 프로세스를 정리
        proc.terminate()
        proc.wait()

    # 프로세스 종료 확인
    proc.wait()

    # Reward 확인 (사용자가 구현)
    reward = get_reward_from_log(log_file)
    return reward

# 초기 상태 설정
state = torch.tensor(np.random.randint(1, 7, num_parameters), dtype=torch.float32).unsqueeze(0)

# 에피소드 반복
num_episodes = 1000
for episode in range(num_episodes):
    action = select_action(state, epsilon)
    new_state = state.clone()
    parameter_index = action // 5
    parameter_value = (action % 5) + 1
    new_state[0][parameter_index] = parameter_value
    
    reward = evaluate_model(new_state)
    reward = torch.tensor([reward], dtype=torch.float32)
    memory.append((state, action, reward, new_state))
    
    state = new_state
    
    optimize_model()
    
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    if epsilon > 0.01:
        epsilon *= 0.995  # 점진적으로 탐색 감소

print("Training completed.")
