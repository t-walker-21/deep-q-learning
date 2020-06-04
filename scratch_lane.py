import numpy as np

for _ in range(10):
    print(np.random.rand())

eps = 0.99
decay = 0.99
true_count = 0
false_count = 0

for _ in range (1000):
    if eps > np.random.rand():
        true_count += 1
        print(True)
    else:
        false_count += 1
        print(False)
    if (eps > 0.1):
        eps *= decay

print("true_count: {}\nfalse_count: {}".format(true_count, false_count))



