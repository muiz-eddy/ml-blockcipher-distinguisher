from Present import encryption
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
    s = bin(int(s,16))[2:].zfill(64)
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
        a = os.urandom(8)
        a = a.hex()
       
        b = hex_bin(a)
        b = flip(b)
        b = binToHexa(b)
        
        pt_list.append(a)
        pt_list.append(b)
    
    f = open("E:\\Data\\Present\\TextFile\\" + user_choice + " " + str(i+1) + ".txt", "w")
    f.write('\n'.join(pt_list))
    f.close()

#READING KEY FILE
key = open("C:\\Users\\mesre\\Desktop\\Final Year Project\\Code\\PRESENT ANALYSIS\\keys.txt", 'r')
keys = key.read().split()

print("Encrypting..." + "\n")
#READING THE PLAINTEXT FILE
for i in range(10):
    file = open("E:\\Data\\Present\\TextFile\\" + user_choice + " " + str(i+1) + ".txt", 'r')
    Lines = file.read().split()
    original = []
    flipped = []
    
    #PRESENT
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
        
        ct = encryption(pt1, key, rounds = 9)   
        # ciphertext_ori = ct.encrypt(pt1)
        # pt = ct.decrypt(ciphertext_ori)
        result_ori_speck.append(hex(ct))
        ct2= encryption(pt2, key, rounds = 9)   
        result_ori_speck.append(hex(ct2))
        
    with open("E:\\Data\\Present\\ctfile_present\\" + user_choice + str(i + 1) + ".txt", "w") as ct1:
        ct1.write("\n".join(result_ori_speck))
            
pd_s = []
pd_a = []
diff_s = []
store = []
chis = []
chia = []

ex = [5.421010862427522e-20, 3.469446951953614e-18, 1.0928757898653885e-16, 2.258609965721803e-15, 3.4443801977257493e-14, 4.133256237270899e-13, 4.064368633316384e-12, 3.367619724747861e-11, 2.399429053882851e-10, 1.4929780779715518e-09, 8.211379428843535e-09, 4.03104081052319e-08, 1.7803763579810755e-07, 7.121505431924302e-07, 2.5942626930581386e-06, 8.647542310193795e-06, 2.64830983249685e-05, 7.477580703520517e-05, 0.00019524794059192462, 0.00047270554038044907, 0.0010635874658560104, 0.00222846897607926, 0.004355643907791281, 0.007953784527271034, 0.013587715234088017, 0.021740344374540827, 0.03261051656181124, 0.04589628256847508, 0.06064865910834207, 0.07528799061725222, 0.08783598905346093, 0.09633624605863457, 0.09934675374796689, 0.09633624605863457, 0.08783598905346093, 0.07528799061725222, 0.06064865910834207, 0.04589628256847508, 0.03261051656181124, 0.021740344374540827, 0.013587715234088017, 0.007953784527271034, 0.004355643907791281, 0.00222846897607926, 0.0010635874658560104, 0.00047270554038044907, 0.00019524794059192462, 7.477580703520517e-05, 2.64830983249685e-05, 8.647542310193795e-06, 2.5942626930581386e-06, 7.121505431924302e-07, 1.7803763579810755e-07, 4.03104081052319e-08, 8.211379428843535e-09, 1.4929780779715518e-09, 2.399429053882851e-10, 3.367619724747861e-11, 4.064368633316384e-12, 4.133256237270899e-13, 3.4443801977257493e-14, 2.258609965721803e-15, 1.0928757898653885e-16, 3.469446951953614e-18, 5.421010862427522e-20]
expected = [i * int(nums) for i in ex]
h = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]

print("Processing Results..." + "\n")

for i in range(10):
    file = open("E:\\Data\\Present\\ctfile_present\\" + user_choice  + str(i+1) + ".txt", "r")
    text = file.read().split()
    
    diff = []
    ori = []
    flip = []
    
    observed_value_s = [0] * 64
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
    with open("E:\\Data\\Present\\result_present\\" + user_choice + " " + str(i + 1) + ".csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        
        # write the header
        writer.writerow(header)
        
        # write the data
        for row in rows:
            writer.writerow(row)

chiavg = chisquare10setavg(chis)
dof = stats.chi2.ppf(1-0.05,df = 64)

print("SPECK Chisquare Average: ", chiavg)
print("Critical Value: ", dof)         

# rows2 = zip(nums,str(chiavg),str(time.process_time() - start))
with open("E:\\Data\\Present\\hypothesis\\" + hypo + ".csv", 'a', encoding='UTF8',newline="") as ct2:
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

