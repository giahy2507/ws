__author__ = 'HyNguyen'


if __name__ == "__main__":
    file_1 = "data/vcl.pre.txt"
    file_2 = "/Users/HyNguyen/Documents/Boomerang/boomerangproject/datatrainword2vec.txt"

    fo = open("data/datatrainword2vec.txt", mode="w")

    with open(file_2, mode="r") as f:
        for i,line in enumerate(f):
            if i % 100000 == 0 and i is not 0:
                print("Process line: ", i)
            line = line.replace("_"," ")
            fo.write(line)

    with open(file_1, mode="r") as f:
        for i,line in enumerate(f):
            if i % 100000 == 0 and i is not 0:
                print("Process line: ", i)
            line = line.replace("_"," ")
            fo.write(line)

    fo.close()
