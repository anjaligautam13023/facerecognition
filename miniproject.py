from create_dataset import *
from opencv import *

if __name__ == '__main__':
    print("welcome to anjali face recognition project")
    print("choose one option")
    print("1-create database")
    print("2-detect face")
    x=int(input())
    if x==1:
        createdataset()
    if x==2:
        x=detect_face()
        #print(x)
        if(x==0):
            print("you are not allowed")
        else:
            print("hlo "+x+" you belongs to this class")
            print("you are most welcome")