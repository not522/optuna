import matplotlib.pyplot as plt

y = range(100)

for dim in range(2, 11):
    for branch in ("master", "fix"):
        values = []
        with open(f"{branch}_{dim}.txt") as f:
            for line in f.readlines():
                values.append(float(line))
        values.sort()
        plt.plot(values, y, label=branch)
    plt.title(f"{dim}-D Sphere")
    plt.xlabel("Objective Value")
    plt.ylabel("Number of Trials")
    plt.legend()
    plt.savefig(f"{dim}.png")
    plt.clf()
