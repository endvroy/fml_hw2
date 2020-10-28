import random

with open('abalone.train') as f:
    lines = f.readlines()

random.shuffle(lines)
lines = lines[:-3]
total = len(lines)
chunk = total // 10

for i in range(10):
    lines_copy = lines.copy()
    val_lines = lines_copy[chunk * i: chunk * (i + 1)]
    del lines_copy[chunk * i: chunk * (i + 1)]
    train_lines = lines_copy
    assert len(train_lines) + len(val_lines) == total

    with open(f'abalone.train.{i}', 'w') as f:
        f.writelines(train_lines)

    with open(f'abalone.val.{i}', 'w') as f:
        f.writelines(val_lines)
