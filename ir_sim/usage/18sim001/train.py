import torch
from sim001_env import MultiRobotSearchEnv
from ppo_trainer import PPO, RolloutBuffer

def train():
    ####### 하이퍼파라미터 설정 #######
    env_name = "MultiRobotSearch_v1"
    max_training_steps = 1000000
    update_timestep = 2000      # 2000스텝마다 정책 업데이트
    K_epochs = 40               # 업데이트 당 반복 횟수
    eps_clip = 0.2              # PPO 클리핑 범위
    gamma = 0.99                # 할인율
    lr_actor = 0.0003           # Actor 학습률
    lr_critic = 0.001           # Critic 학습률
    ################################

    env = MultiRobotSearchEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    buffer = RolloutBuffer()

    time_step = 0
    i_episode = 0

    while time_step <= max_training_steps:
        state, _ = env.reset()
        current_ep_reward = 0
        
        for t in range(1, 1001): # 최대 1000스텝
            # 1. 행동 선택            
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            action, action_logprob, state_val = ppo_agent.policy_old.act(state_tensor)
            
            
            # 3. 중요: 시뮬레이터용으로 형태 변환 (1, 2) -> (2,)
            # squeeze()를 하면 [[v, w]]가 [v, w]로 바뀝니다.
            env_action = action.squeeze().numpy() 
            
            # 4. 수정 전: env.step([action.numpy(), action.numpy()]) <- 여기서 에러 발생
            # 수정 후: env_action 변수를 리스트에 담아 전달
            obs, reward, terminated, truncated, _ = env.step([env_action, env_action])
    
    
            # 3. 데이터 저장
            buffer.states.append(state_tensor.squeeze(0)) # 저장할 때는 다시 (32,)로
            buffer.actions.append(action.squeeze(0))
            buffer.logprobs.append(action_logprob)
            buffer.rewards.append(reward)
            buffer.is_terminals.append(terminated)
            
            time_step += 1
            current_ep_reward += reward
            state = obs

            # 업데이트 주기 확인
            if time_step % update_timestep == 0:
                ppo_agent.update(buffer)
                buffer.clear()

            if terminated or truncated:
                break
        
        i_episode += 1
        if i_episode % 10 == 0:
            print(f"Episode: {i_episode} \t Steps: {time_step} \t Reward: {current_ep_reward:.2f}")
            
        # 모델 저장
        if i_episode % 100 == 0:
            torch.save(ppo_agent.policy.state_dict(), "ppo_model_latest.pth")

if __name__ == '__main__':
    train()