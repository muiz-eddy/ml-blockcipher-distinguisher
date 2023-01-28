from speck import SpeckCipher
import numpy as np
import scipy.stats as stats
import os
import random
import csv
import time

#BIT DIFFERENCE
def solve(A, B):
    XOR = A ^ B
    count = 0
    while (XOR):
        XOR = XOR & (XOR - 1)
        count += 1
    return count

def chisquare(observed_val, expected):
    chi = []
    sums = 0
    for i in range (len(observed_val)):
        calc = (observed_val[i] - expected[i]) * (observed_val[i] - expected[i]) / expected[i]
        chi.append(calc)
        
        if expected[i] > 5.0:
            calc = (observed_val[i] - expected[i]) * (observed_val[i] - expected[i]) / expected[i]
            sums = sums + calc
            
        else:
            calc = 0
            sums = sums + calc
    return sums,chi

def flip(s):
    pos = random.SystemRandom().choice(range(len(s)))
    r = list(s)
    
    r[pos] = '1' if r[pos] == '0' else '0'
    
    return ''.join(r)

def hex_bin(s):
    s = bin(int(s,16))[2:].zfill(32)
    return s

def binToHexa(n):
    # convert int to hexadecimal
    hex_num = '{:0{}X}'.format(int(n, 2), len(n) //4)
    return(hex_num)

def chisquare10setavg(chi):
    b = 0
    for i in range(10):
        a = np.sum(chi[i])
        b += a
        
    b = b/10
    return b


user_choice = input("File name: ")
hypo = input ("hypo: ")
nums = input("Number: ")
#SPECK
start = time.process_time()
print("Generating Random Plaintext..." + "\n")
for i in range(10):
    pt_list = []
    pt_list2 = []
    
    for j in range(int(nums)):
        a = os.urandom(4)
        a = a.hex()
       
        b = hex_bin(a)
        b = flip(b)
        b = binToHexa(b)
        
        pt_list.append(a)
        pt_list.append(b)
    
    f = open("E:\\Data\\Speck_64_32\\TextFile\\" + user_choice + " " + str(i+1) + ".txt", "w")
    f.write('\n'.join(pt_list))
    f.close()

#READING KEY FILE
key = open("C:\\Users\\mesre\\Desktop\\Final Year Project\\Code\\SPECK ANALYSIS\\keys.txt", 'r')
keys = key.read().split()

print("Encrypting..." + "\n")
#READING THE PLAINTEXT FILE
for i in range(10):
    file = open("E:\\Data\\Speck_64_32\\TextFile\\" + user_choice + " " + str(i+1) + ".txt", 'r')
    Lines = file.read().split()
    original = []
    flipped = []
    
    #SPECK 
    result_ori_speck = []
      
    #APPENDING PLAINTEXT TO 2 DIFFERENT LIST
    for j in range (len(Lines)):
        if j % 2 != 0:
            flipped.append(Lines[j])
            
        else:
            original.append(Lines[j])
    
    #SPECK ENCRYPTION 
    for j in range(len(original)):
        key = int(keys[i],16)
        pt1 = int(original[j],16)
        pt2 = int(flipped[j],16)
        
        ct = SpeckCipher(key, 64, 32, 'ECB')    
        ciphertext_ori = ct.encrypt(pt1)
        
        result_ori_speck.append(format(ciphertext_ori,'x'))
        ciphertext_flip = ct.encrypt(pt2)
        result_ori_speck.append(format(ciphertext_flip,'x'))
        
    with open("E:\\Data\\Speck_64_32\\ctfile_speck\\" + user_choice + str(i + 1) + ".txt", "w") as ct1:
        ct1.write("\n".join(result_ori_speck))
            
pd_s = []
pd_a = []
diff_s = []
store = []
chis = []
chia = []

ex = [2.3283064365386963e-10, 7.450580596923828e-09, 1.1548399925231934e-07, 1.1548399925231934e-06, 8.372589945793152e-06, 4.688650369644165e-05, 0.00021098926663398743, 0.000783674418926239, 0.002448982559144497, 0.0065306201577186584, 0.015020426362752914, 0.03004085272550583, 0.0525714922696352, 0.08087921887636185, 0.10976465418934822, 0.13171758502721786, 0.13994993409141898, 0.13171758502721786, 0.10976465418934822, 0.08087921887636185, 0.0525714922696352, 0.03004085272550583, 0.015020426362752914, 0.0065306201577186584, 0.002448982559144497, 0.000783674418926239, 0.00021098926663398743, 4.688650369644165e-05, 8.372589945793152e-06, 1.1548399925231934e-06, 1.1548399925231934e-07, 7.450580596923828e-09, 2.3283064365386963e-10]
expected = [i * int(nums) for i in ex]
h = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

print("Processing Results..." + "\n")

for i in range(10):
    file = open("E:\\Data\\Speck_64_32\\ctfile_speck\\" + user_choice  + str(i+1) + ".txt", "r")
    text = file.read().split()
    
    diff = []
    ori = []
    flip = []
    
    observed_value_s = [0] * 32
    for j in range(len(text)):
        if j % 2 != 0:
            flip.append(text[j])
            
        else:
            ori.append(text[j])
            
    #CHECKING FOR THE BIT DIFFERENCES
    for j in range(len(ori)):
        test = solve(int(ori[j],16),int(flip[j],16))
        diff.append(test)
    diff_s.append(diff)
    for j in range(len(observed_value_s)):
        for z in range(len(diff)):
            if j == diff[z]:
                observed_value_s[j] += 1
    pd_s.append(observed_value_s)       
    
   

for i in range(len(pd_s)):
    chi,chi2= chisquare(pd_s[i],expected)
    chis.append(chi2)
    
 
for i in range(10):
    header = ['Position', 'Total Bits Change (observed)', 'Expected (Binomial (n = 128, p = 0.5)', 'Chisquare']
    
    rows = zip(h,pd_s[i],expected,chis[i])
    with open("E:\\Data\\Speck_64_32\\result_speck\\" + user_choice + " " + str(i + 1) + ".csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        
        # write the header
        writer.writerow(header)
        
        # write the data
        for row in rows:
            writer.writerow(row)

chiavg = chisquare10setavg(chis)
dof = stats.chi2.ppf(1-0.05,df = 32)

print("SPECK Chisquare Average: ", chiavg)
print("Critical Value: ", dof)         

# rows2 = zip(nums,str(chiavg),str(time.process_time() - start))
with open("E:\\Data\\Speck_64_32\\hypothesis\\" + hypo + ".csv", 'a', encoding='UTF8',newline="") as ct2:
        header2 = ['#Text', 'Average', 'Performance(s)']
        writer = csv.writer(ct2)
        writer2 = csv.DictWriter(ct2, delimiter=',', lineterminator='\n',fieldnames=header2)
        
        if (ct2.tell()) == 0:
            writer2.writeheader()  # file doesn't exist yet, write a header
        
        writer.writerow([nums,str(chiavg),str(str(time.process_time() - start))])
ct2.close()      
ct2.close()      
print("Process time: ",time.process_time() - start, "\n")  

print("Results saved.")
# print("Speck Chisquare Average: ", chisquare10setavg(chis))
# print("Critical Value: ", stats.chi2.ppf(1-0.05,df = 127))

# r_values = list(range(n))
# plt.bar(r_values, chis[0])
# plt.show()

