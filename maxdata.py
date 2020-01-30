from collections import OrderedDict
path = "data/"
filename = "data"
if __name__ == "__main__":
    data = OrderedDict()
    # Read all data files and take the max for each matrix size
    for i in range(1, 11):
        file = open(path + filename + str(i) + ".txt")
        print("Opened data" + str(i) + ".txt")
        line = file.readline()
        while line:
            n, speed = line.split("\t")
            speed = float(speed.split("\n")[0])
            if n not in data:
                data[n] = speed
            else:
                data[n] = max(data[n], speed)
            line = file.readline()
        file.close()
    # Print max data to file
    file = open("data.txt", "w")
    for n in data:
        file.write(str(n) + "\t" + str(data[n]) + "\n")
    file.close()
    sum = 0
    for n in data:
        sum += data[n]
    print("Average: " + str(sum/len(data)))
