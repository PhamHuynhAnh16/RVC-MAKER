import os
import sys
import time
import logging
import webbrowser

from tensorboard import program

sys.path.append(os.getcwd())

from main.configs.config import Config

config = Config()
translations = config.translations

def launch_tensorboard():
    for l in ["root", "tensorboard"]:
        logging.getLogger(l).setLevel(logging.ERROR)

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", config.configs["logs_path"], f"--port={config.configs['tensorboard_port']}"])
    url = tb.launch()

    print(f"{translations['tensorboard_url']}: {url}")
    if "--open" in sys.argv: webbrowser.open(url)

    return f"{translations['tensorboard_url']}: {url}"

if __name__ == "__main__": 
    launch_tensorboard()

    while 1:
        time.sleep(5)