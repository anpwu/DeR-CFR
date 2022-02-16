import numpy as np
from generator.data_generator import run as generator_run
import json

with open('run.json', 'r') as f:
    run_dict = json.load(f)

print(run_dict["generator_16_16_16"])

generator_run(run_dict["generator_16_16_16"], 3000)


