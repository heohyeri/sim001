import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim=32, action_dim=2, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # 1. 공통 특징 추출 레이어 (Shared Feature Extractor)
        # LiDAR 데이터와 위치 정보를 처리하는 기본 층입니다.
        self.affine = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. Actor (Policy Network): 어떤 행동을 할지 결정
        # 선속도(v)와 각속도(w)의 평균(mu)과 표준편차(sigma)를 출력합니다.
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Parameter(torch.zeros(action_dim))
        
        # 3. Critic (Value Network): 현재 상태의 가치를 평가
        # 이 상태가 보상을 받기에 얼마나 유리한지 점수(Value)를 매깁니다.]
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        # 상태 정보를 공통 레이어에 통과시킵니다.
        x = self.affine(state)
        
        # Actor 로직
        mu = self.mu_layer(x)
        # 선속도는 0~1.0, 각속도는 -1.0~1.0 범위를 가지도록 tanh 사용
        mu = torch.tanh(mu) 
        
        std = torch.exp(self.log_std_layer).expand_as(mu)
        dist = Normal(mu, std) # 가우시안 분포 생성
        
        # Critic 로직
        value = self.value_layer(x)
        
        return dist, value

    def act(self, state):
        # 학습 중 행동을 선택할 때 사용 (샘플링)
        dist, value = self.forward(state)
        action = dist.sample()
        # 액션 범위를 시뮬레이션 제한값에 맞게 클리핑
        action = torch.clamp(action, -1.0, 1.0)
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        return action.detach(), action_logprob.detach(), value.detach()

    def evaluate(self, state, action):
        # 업데이트 시 현재 정책 하에서의 행동 확률과 가치를 재계산
        dist, value = self.forward(state)
        action_logprob = dist.log_prob(action).sum(axis=-1)
        dist_entropy = dist.entropy().sum(axis=-1)
        
        return action_logprob, value, dist_entropy