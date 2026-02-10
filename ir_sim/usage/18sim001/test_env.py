import numpy as np
from sim001_env import MultiRobotSearchEnv

def test_random_agent():
    # 1. 환경 인스턴스 생성
    env = MultiRobotSearchEnv()
    
    episodes = 3 # 테스트할 에피소드 횟수
    print("Starting Random Action Test...")

    for ep in range(episodes):
        # 2. 환경 초기화
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        print(f"\n--- Episode {ep+1} Start ---")
        
        while not done:
            # 3. 무작위 행동 선택 (선속도 v, 각속도 w)
            # 로봇 2대를 위해 각각의 랜덤 액션 생성
            action0 = env.action_space.sample()
            action1 = env.action_space.sample()
            action_list = [action0, action1]
            
            # 4. 환경 실행 (Step)
            obs, reward, terminated, truncated, info = env.step(action_list)
            
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            
            # 5. 실시간 렌더링
            env.render()
            
            # 100 스텝마다 로그 출력
            if step_count % 100 == 0:
                print(f"Step: {step_count} | Last Reward: {reward:.2f} | Total: {total_reward:.2f}")
            
            # 테스트를 위해 너무 오래 걸리면 강제 종료
            if step_count > 2000:
                print("Episode Truncated (Max Steps reached)")
                break
        
        print(f"Episode {ep+1} Finished | Total Steps: {step_count} | Total Reward: {total_reward:.2f}")

    print("\nAll Tests Completed Successfully!")

if __name__ == "__main__":
    test_random_agent()