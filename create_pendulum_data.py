import gym
import numpy as np

def create_pendulum_datasets():
    env = gym.make('Pendulum-v1')
    num_train = 140000
    num_val_test = 28000
    nsteps = 5
    x_dim = 784  # The state space dimension we want to match
    a_dim = 1
    r_dim = 1
    mask_dim = 1
    u_dim = 1  # Assuming u_dim for rich_train

    def collect_data(num_samples):
        states = []
        actions = []
        rewards = []
        masks = []
        rich = []

        for _ in range(num_samples):
            state = env.reset()
            ep_states = []
            ep_actions = []
            ep_rewards = []
            ep_masks = []
            ep_rich = []

            for _ in range(nsteps):
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                
                # Flatten the state to fit the required dimension
                flat_state = np.resize(state, (x_dim,))
                
                ep_states.append(flat_state)
                ep_actions.append([action])
                ep_rewards.append([reward])
                ep_masks.append([1.0])
                ep_rich.append([1.0])  # Dummy value for rich

                state = next_state

            states.append(ep_states)
            actions.append(ep_actions)
            rewards.append(ep_rewards)
            masks.append(ep_masks)
            rich.append(ep_rich)

        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), np.array(rewards, dtype=np.float32), np.array(masks, dtype=np.float32), np.array(rich, dtype=np.float32)

    # Collect training, validation, and testing data
    x_train, a_train, r_train, mask_train, rich_train = collect_data(num_train)
    x_validation, a_validation, r_validation, mask_validation, rich_validation = collect_data(num_val_test)
    x_test, a_test, r_test, mask_test, rich_test = collect_data(num_val_test)

    # Save datasets as npz files
    np.savez('pendulum_training_data.npz', x_train=x_train, a_train=a_train, r_train=r_train, mask_train=mask_train, rich_train=rich_train)
    np.savez('pendulum_validation_data.npz', x_validation=x_validation, a_validation=a_validation, r_validation=r_validation, mask_validation=mask_validation, rich_validation=rich_validation)
    np.savez('pendulum_testing_data.npz', x_test=x_test, a_test=a_test, r_test=r_test, mask_test=mask_test, rich_test=rich_test)

    print("Pendulum datasets created and saved.")

create_pendulum_datasets()
