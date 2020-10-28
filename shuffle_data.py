import random

with open('abalone.train') as f:
    lines = f.readlines()

random.shuffle(lines)

with open(f'abalone.train.shuffled', 'w') as f:
    f.writelines(lines)
