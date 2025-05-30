import matplotlib.pyplot as plt

for dim in range(2, 11):
    for branch in ("master", "fix"):
        values = []
        with open(f"{branch}_{dim}.txt") as f:
            for line in f.readlines():
                values.append(float(line))
        values.sort()
        plt.plot(values, range(100), label=branch)
    plt.title(f"{dim}-D Sphere")
    plt.xlabel("Objective Value")
    plt.ylabel("Number of Trials")
    plt.legend()
    plt.savefig(f"{dim}.png")
    plt.clf()
