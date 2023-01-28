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
    s = bin(int(s,16))[2:].zfill(128)
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
    
    f = open("E:\\Data\\Speck\\TextFile\\" + user_choice + " " + str(i+1) + ".txt", "w")
    f.write('\n'.join(pt_list))
    f.close()

#READING KEY FILE
key = open('C:\\Users\\mesre\\Desktop\\Final Year Project\\Code\\SPECK ANALYSIS\\key.txt', 'r')
keys = key.read().split()

print("Encrypting..." + "\n")
#READING THE PLAINTEXT FILE
for i in range(10):
    file = open("E:\\Data\\Speck\\TextFile\\" + user_choice + " " + str(i+1) + ".txt", 'r')
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
        
        ct = SpeckCipher(key, 128, 128, 'ECB')    
        ciphertext_ori = ct.encrypt(pt1)
        pt = ct.decrypt(ciphertext_ori)
        result_ori_speck.append(format(ciphertext_ori,'x'))
        ciphertext_flip = ct.encrypt(pt2)
        result_ori_speck.append(format(ciphertext_flip,'x'))
        
    with open("E:\\Data\\Speck\\ctfile_speck\\" + user_choice + str(i + 1) + ".txt", "w") as ct1:
        ct1.write("\n".join(result_ori_speck))
            
pd_s = []
pd_a = []
diff_s = []
store = []
chis = []
chia = []

ex = [2.938735877055719e-39, 3.76158192263132e-37, 2.3886045208708882e-35, 1.003213898765773e-33, 3.135043433643041e-32, 7.774907715434741e-31, 1.593856081664122e-29, 2.7778634566146125e-28, 4.2015184781296014e-27, 5.602024637506135e-26, 6.666409318632301e-25, 7.151239087260105e-24, 6.972458110078602e-23, 6.221578005916291e-22, 5.110581933431239e-21, 3.884042269407742e-20, 2.7431048527692177e-19, 1.807222020647955e-18, 1.1144535793995723e-17, 6.45209967020805e-17, 3.5163943202633876e-16, 1.8084313647068848e-15, 8.79555254652894e-15, 4.053602477965512e-14, 1.7734510841099112e-13, 7.377556509897231e-13, 2.922647386613134e-12, 1.1041112349427394e-11, 3.982686954614882e-11, 1.3733403291775454e-10, 4.5320230862858996e-10, 1.4327040724387683e-09, 4.342884219580017e-09, 1.2633845002414594e-08, 3.530044927145254e-08, 9.480692090047253e-08, 2.449178789928874e-07, 6.08984996414747e-07, 1.4583588072037363e-06, 3.3654434012393916e-06, 7.488111567757646e-06, 1.6072044340552996e-05, 3.3292091848288346e-05, 6.658418369657669e-05, 0.00012862853668656863, 0.0002401066018149281, 0.000433235825013892, 0.0007558582478965775, 0.0012755107933254746, 0.0020824666013477136, 0.0032902972301293875, 0.0050322192931390635, 0.007451555491763613, 0.010685249384415747, 0.01484062414502187, 0.019967385213302154, 0.026028912867340305, 0.03287862677979828, 0.040247974161477205, 0.04775183375090516, 0.05491460881354093, 0.061216285234766944, 0.06615308243111911, 0.06930322921355336, 0.07038609217001514, 0.06930322921355336, 0.06615308243111911, 0.061216285234766944, 0.05491460881354093, 0.04775183375090516, 0.040247974161477205, 0.03287862677979828, 0.026028912867340305, 0.019967385213302154, 0.01484062414502187, 0.010685249384415747, 0.007451555491763613, 0.0050322192931390635, 0.0032902972301293875, 0.0020824666013477136, 0.0012755107933254746, 0.0007558582478965775, 0.000433235825013892, 0.0002401066018149281, 0.00012862853668656863, 6.658418369657669e-05, 3.3292091848288346e-05, 1.6072044340552996e-05, 7.488111567757646e-06, 3.3654434012393916e-06, 1.4583588072037363e-06, 6.08984996414747e-07, 2.449178789928874e-07, 9.480692090047253e-08, 3.530044927145254e-08, 1.2633845002414594e-08, 4.342884219580017e-09, 1.4327040724387683e-09, 4.5320230862858996e-10, 1.3733403291775454e-10, 3.982686954614882e-11, 1.1041112349427394e-11, 2.922647386613134e-12, 7.377556509897231e-13, 1.7734510841099112e-13, 4.053602477965512e-14, 8.79555254652894e-15, 1.8084313647068848e-15, 3.5163943202633876e-16, 6.45209967020805e-17, 1.1144535793995723e-17, 1.807222020647955e-18, 2.7431048527692177e-19, 3.884042269407742e-20, 5.110581933431239e-21, 6.221578005916291e-22, 6.972458110078602e-23, 7.151239087260105e-24, 6.666409318632301e-25, 5.602024637506135e-26, 4.2015184781296014e-27, 2.7778634566146125e-28, 1.593856081664122e-29, 7.774907715434741e-31, 3.135043433643041e-32, 1.003213898765773e-33, 2.3886045208708882e-35, 3.76158192263132e-37, 2.938735877055719e-39]
expected = [i * int(nums) for i in ex]
h = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]

print("Processing Results..." + "\n")

for i in range(10):
    file = open("E:\\Data\\Speck\\ctfile_speck\\" + user_choice  + str(i+1) + ".txt", "r")
    text = file.read().split()
    
    diff = []
    ori = []
    flip = []
    
    observed_value_s = [0] * 128
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
    with open("E:\\Data\\Speck\\result_speck\\" + user_choice + " " + str(i + 1) + ".csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        
        # write the header
        writer.writerow(header)
        
        # write the data
        for row in rows:
            writer.writerow(row)

chiavg = chisquare10setavg(chis)
dof = stats.chi2.ppf(1-0.01,df = 127)

print("SPECK Chisquare Average: ", chiavg)
print("Critical Value: ", dof)         

# rows2 = zip(nums,str(chiavg),str(time.process_time() - start))
with open("E:\\Data\\Speck\\hypothesis\\" + hypo + ".csv", 'a', encoding='UTF8',newline="") as ct2:
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

