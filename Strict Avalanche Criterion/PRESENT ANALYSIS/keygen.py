import os

key = []
for i in range(10):
    rand = os.urandom(10)
    key.append(rand.hex())
    
with open("keys.txt", "w") as txt:
    for i in key:
        txt.write("".join(i) + "\n")