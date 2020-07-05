from kaggle_environments import evaluate, make
import numpy as np
import torch
import os
import logging
import time

from model.model import DQNConvAgent
from utils.utils import learn_image

# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(filename)s:%(funcName)s:\n\t%(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(filename)s:%(funcName)s:\t\t%(message)s')
if torch.cuda.is_available():
    device = 'cuda'
    logging.info("cuda is available")

else:
    device = 'cpu'
    logging.info("cuda is not available")

checkpoint_dir = "checkpoints/connect_x/"
target_net = None


def convert_markers(tensor_stat, mark_1, mark_n1):
    tensor_stat[tensor_stat == mark_n1] = -1
    tensor_stat[tensor_stat == mark_1] = 1


def average_list(lst):
    return sum(lst) / len(lst)


def mean_reward(rewards, player):
    logging.debug(rewards)
    return sum(none_to_neg_1(r[player]) for r in rewards) / float(len(rewards))


def none_to_neg_1(item):
    if item is None:
        return -1
    else:
        return item


def get_opposing_mark(my_marker):
    if my_marker == 1:
        opposing_mark = 2
    elif my_marker == 2:
        opposing_mark = 1
    else:
        logging.warning("Error assigning opponent mark")
        opposing_mark = None
    return opposing_mark


# class MyHelper:
#     def __init__(self, my_agent=None):
#         self.my_agent = my_agent
#
#     # def move_helper(self, observation, configuration):
#     #     logging.debug(f"observation: {observation}")
#     #     logging.debug(f"configuration: {configuration}")
#
#     def move_helper(self, structure=None, woot=None, another=None):
#         logging.debug(f"self: {self}")
#         # for i, arg in enumerate(args):
#         #     logging.debug(f"args {i}: {arg}")
#         logging.debug(f"structure: {structure}")
#         logging.debug(f"woot: {woot}")
#         # logging.debug(f"structure['board']: {structure['board']}")
#         # logging.debug(f"structure['mark']: {structure['mark']}")

def opposing_agent(observation, configuration):
    # logging.debug(f"observation: {observation}")
    # logging.debug(f"configuration: {configuration}")
    # logging.debug(f"target_net: {target_net}")
    my_board = observation.board
    my_mark = observation.mark
    op_mark = get_opposing_mark(my_mark)
    rows = configuration.rows
    cols = configuration.columns
    my_board_tensor = torch.Tensor(my_board).reshape(1, rows, cols).to(device)
    convert_markers(my_board_tensor, my_mark, op_mark)  # makes in place modification
    action = agent.choose_action(my_board_tensor)
    return action

env = make("connectx", debug=True)
logging.info("Render initial board")
env.render()
# Play as first position against opposing agent.
# trainer = env.train([None, "random"])
trainer = env.train([None, opposing_agent])
state_struct = trainer.reset()

configuration = env.configuration
# Not needed.
logging.debug(f"state_struct.mark: {state_struct.mark}, len(state_struct.board): {len(state_struct.board)}, "
              f"type(state_struct.board): {type(state_struct.board)}, \n\tstate_struct.board: {state_struct.board}")
logging.info("Rows: {}".format(configuration.rows))
logging.info("Columns: {}".format(configuration.columns))
rows = configuration.rows
cols = configuration.columns
board_shape = (rows, cols)
my_mark = state_struct.mark
op_mark = get_opposing_mark(my_mark)

rows = configuration.rows
state_space = (1,) + board_shape  # (C, H, W)

# Create training and target net.
agent = DQNConvAgent(state_space=state_space, action_space=cols, decay=0.995).to(device)
target_net = DQNConvAgent(state_space=state_space, action_space=cols, buf_len=1000000).to(device)
target_net.load_state_dict(agent.state_dict())

criterion = torch.nn.MSELoss()
opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
batch_size = 256

iteration_count = 0
total_wins = 0

# reward_list = list()
# global thread_var
# thread_var = False
#
# x = threading.Thread(target=get_user_input)
# x.start()

# Create checkpoints folder if it doesn't exist.
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

while True:
    recent_wins = list()
    # Fill the replay buffer
    logging.info(f"*********************** Populate Replay Buffer ***********************")
    agent.train()
    while not agent.replay_memory.is_ready():
        logging.info(f"Game iteration count: {iteration_count}")
        logging.debug(f"Reset Board")
        state_struct = trainer.reset()
        # logging.debug(f"type(state): {type(state_struct)}")
        state = state_struct['board']
        state_tensor = torch.Tensor(state).reshape(1, rows, cols).to(device)

        # if iteration_count % 50 == 0:



            # logging.info(max_values)

        while not env.done:
            # logging.debug(state_struct)
            # logging.debug(type(state_struct))
            # logging.debug(type(state_struct['board']))
            # logging.debug(state_struct['board'])
            logging.debug(f"state_tensor: \n{state_tensor}")
            # Set opponent markers to -1, friendly marker to 1
            convert_markers(state_tensor, my_mark, op_mark)  # makes in place modification
            # logging.debug(f"state_tensor: \n{state_tensor}")
            action = agent.choose_action(state_tensor)
            logging.debug(f"action: {action}")
            next_state_struct, reward, done, info = trainer.step(action)
            # logging.debug(f"next_state_struct: {next_state_struct}\nreward: {reward}\ndone: {done}\ninfo: {info}")
            # logging.debug(f"env.done: {env.done}")
            next_state = next_state_struct["board"]
            next_state_tensor = torch.Tensor(next_state).reshape(1, rows, cols).to(device)
            # logging.debug(type(next_state_tensor))
            done = int(done)
            if reward is None:
                reward = -5

            if reward not in (-5, -1, 0, 1):
                logging.warning(f"reward: {reward}")
            agent.store_experience((state_tensor, action, reward, next_state_tensor, done))
            state_tensor = next_state_tensor
        if reward < 0:
            recent_wins.append(-1)
        else:
            recent_wins.append(reward)
        logging.debug(f"Game iteration over.  reward: {reward}")
        logging.debug(f"next_state_tensor: \n{next_state_tensor}")

        iteration_count += 1
        agent.anneal_eps()
        logging.info("Ending reward: {}".format(reward))

    logging.info(f"Recent win percent: {average_list(recent_wins) * 100}%")
    # logging.info(f"Total average reward: {average_list(reward_list)}")

    # Learn from replay buffer
    logging.info(f"*********************** Begin reinforcement learning ***********************")
    for _ in range(200):
        # learn_image(agent, target, opt, criterion, batch_size, device, image_shape=(100,100)):
        learn_image(agent, target_net, opt, criterion, batch_size, device, (rows, cols))

    logging.info(f"*********************** Saving model ***********************")
    torch.save(agent.state_dict(), checkpoint_dir + "model_" + str(iteration_count) + ".pt")

    logging.info("*********************** Updating target network ***********************")
    target_net.load_state_dict(agent.state_dict())
    logging.info(f"espilon: {agent.eps}")

    logging.info(f"*********************** Evaluation play ***********************")
    agent.eval()
    output = mean_reward(evaluate("connectx", [opposing_agent, "random"], num_episodes=100), player=0)
    logging.info(f"My Agent vs Random Agent: {output}")
    time.sleep(2)

    # Drop some of the replay buffer
    logging.info(f"*********************** Drop items from replay buffer ***********************")
    agent.replay_memory.drop_buffer(percent=0.05)













board_np = np.zeros((configuration.rows, configuration.columns)).astype(int)

