from kaggle_environments import evaluate, make
import numpy as np
import torch
import os
import logging

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


def convert_markers(tensor_stat, mark_1, mark_n1):
    tensor_stat[tensor_stat == mark_n1] = -1
    tensor_stat[tensor_stat == mark_1] = 1

env = make("connectx", debug=True)
logging.info("Render initial board")
env.render()
# Play as first position against opposing agent.
trainer = env.train([None, "random"])
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
if my_mark == 1:
    op_mark = 2
elif my_mark == 2:
    op_mark = 1
else:
    logging.warning("Error assigning opponent mark")
    op_mark = None

rows = configuration.rows
state_space = (1,) + board_shape  # (C, H, W)
agent = DQNConvAgent(state_space=state_space, action_space=cols).to(device)
target_net = DQNConvAgent(state_space=state_space, action_space=cols, buf_len=1000000).to(device)
target_net.load_state_dict(agent.state_dict())

criterion = torch.nn.MSELoss()
opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
batch_size = 256

iteration_count = 0
# global thread_var
# thread_var = False
#
# x = threading.Thread(target=get_user_input)
# x.start()

# Create checkpoints folder if it doesn't exist.
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

while True:

    # Fill the replay buffer
    while not agent.replay_memory.is_ready():
        state_struct = trainer.reset()
        logging.debug(f"type(state): {type(state_struct)}")
        state = state_struct['board']
        state_tensor = torch.Tensor(state).reshape(1, rows, cols).to(device)

        if iteration_count % 50 == 0:
            logging.info(f"Game iteration count: {iteration_count}")
            logging.info("updating target network")
            target_net.load_state_dict(agent.state_dict())
            logging.info(f"espilon: {agent.eps}")
            # logging.info(max_values)

        if iteration_count % 20 == 0:
            logging.info("Saving model")
            torch.save(agent.state_dict(), checkpoint_dir + "model_" + str(iteration_count) + ".pt")

        while not env.done:
            # logging.debug(state_struct)
            # logging.debug(type(state_struct))
            # logging.debug(type(state_struct['board']))
            # logging.debug(state_struct['board'])
            logging.debug(f"state_tensor: {state_tensor}")
            # Set opponent markers to -1, friendly marker to 1
            convert_markers(state_tensor, my_mark, op_mark)  # makes in place modification
            logging.debug(f"state_tensor: {state_tensor}")
            action = agent.choose_action(state_tensor)
            logging.debug(f"action: {action}")
            next_state_struct, reward, done, info = trainer.step(action)
            logging.debug(f"next_state_struct: {next_state_struct}\nreward: {reward}\ndone: {done}\ninfo: {info}")
            logging.debug(f"env.done: {env.done}")
            next_state = next_state_struct["board"]
            next_state_tensor = torch.Tensor(next_state).reshape(1, rows, cols).to(device)
            logging.debug(type(next_state_tensor))
            done = int(done)

            agent.store_experience((state_tensor, action, reward, next_state_tensor, done))
            state_tensor = next_state_tensor

        iteration_count += 1
        agent.anneal_eps()
        logging.info("Ending reward: {}".format(reward))

    # Learn from replay buffer
    for _ in range(200):
        # learn_image(agent, target, opt, criterion, batch_size, device, image_shape=(100,100)):
        learn_image(agent, target_net, opt, criterion, batch_size, device, (rows, cols))

    # Drop some of the replay buffer
    agent.replay_memory.drop_buffer(percent=0.75)













board_np = np.zeros((configuration.rows, configuration.columns)).astype(int)

