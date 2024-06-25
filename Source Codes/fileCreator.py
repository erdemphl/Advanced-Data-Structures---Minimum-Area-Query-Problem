import numpy as np

with open("input.txt", "w") as file:
    file.write(str(5) + " " + str(6) + "\n")
    for i in range(5):
        for j in range(6):
            file.write(str(np.random.randint(50000)) + " ")
        file.write("\n")

