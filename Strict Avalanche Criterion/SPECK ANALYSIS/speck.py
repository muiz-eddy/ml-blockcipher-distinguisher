#BLOCK SIZE ----> KEY SIZE : NUMBER OF ROUNDS'
#32bits --------> 64       : 22
#48bits --------> 72       : 22
#48bits --------> 96       : 23
#64bits --------> 96       : 26
#96bits --------> 96       : 28
#96bits --------> 144      : 29
#128bits -------> 128      : 32
#128bits -------> 192      : 33
#128bits -------> 256      : 34

#X[0]
#X[1]

class SpeckCipher(object):
    speck_setup = {32: {64: 6},
                   48: {72: 22, 96: 23},
                   64: {96: 26, 128: 27},
                   96: {96: 28, 144: 29},
                   128: {128: 9, 192: 33, 256: 34}}
    
    modes_operations = ['ECB']
    
    def __init__(self, key, key_size = 128, block_size = 128, mode = 'ECB', init = 0, counter = 0):
        
        #BLOCK SIZE AND WORD SIZE
        self.setups = self.speck_setup[block_size] 
        self.block_size = block_size
        self.word_size = self.block_size >> 1
        
        #KEYS AND ROUND
        self.rounds = self.setups[key_size]
        self.key_size = key_size 
        
        if self.block_size == 32:
            self.beta_shift = 2
            self.alpha_shift = 7
            
        else:
            self.beta_shift = 3
            self.alpha_shift = 8
            
        #IV for Mode of operation
        self.iv = init & ((2 ** self.block_size) - 1)
        self.iv_upper = self.iv >> self.word_size
        self.iv_lower = self.iv 
        
        #Counter 
        self.counter = counter & ((2 ** self.block_size) - 1)
        
        #Checking for mode
        position = self.modes_operations.index(mode)
        self.mode = self.modes_operations[position]
        
        #bit masking
        self.mod_mask = (2 ** self.word_size) - 1

        # Mod mask for modular subtraction
        self.mod_mask_sub = (2 ** self.word_size)
        
        #key length checker
        self.key = key & ((2 ** self.key_size) - 1)
        
        self.key_schedule = [self.key & self.mod_mask]
        lschedule = [(self.key >> (x * self.word_size)) & self.mod_mask for x in range(1, self.key_size // self.word_size)]
        
        for i in range(self.rounds - 1):
            new_key = self.encrypt_round(lschedule[i], self.key_schedule[i], i)
            lschedule.append(new_key[0])
            self.key_schedule.append(new_key[1])
        

###########################################################################################################################################################################
###ENCRYPTION     
    def encrypt_round(self, p1, p2, k):
         
         rotatingX1 = ((p1 << (self.word_size - self.alpha_shift)) + (p1 >> self.alpha_shift)) & self.mod_mask

         addition = (rotatingX1 + p2) & self.mod_mask

         x = k ^ addition

         ls_y = ((p2 >> (self.word_size - self.beta_shift)) + (p2 << self.beta_shift))  & self.mod_mask

         y = x ^ ls_y

         return x, y
     
    def encrypting(self,p1,p2):
       
        #ENCRYPTION ACCORDING TO ROUND
        for i in self.key_schedule:
            rotatingX1 = ((p1 << (self.word_size - self.alpha_shift)) + (p1 >> self.alpha_shift)) & self.mod_mask
            
            addition = (rotatingX1 + p2) & self.mod_mask
            
            p1 =  i ^ addition 
            
            ls_y = ((p2 >> (self.word_size - self.beta_shift)) + (p2 << self.beta_shift)) & self.mod_mask
            
            p2 = p1 ^ ls_y
            
        return p1, p2
    
    def encrypt(self,plaintext):
        p2 = (plaintext >> self.word_size) & self.mod_mask
        p1 = plaintext & self.mod_mask
        
        if self.mode == 'CBC':
            p2 ^= self.iv_upper
            p1 ^= self.iv_lower
            p2, a = self.encrypting(p2, p1)
        
            self.iv_upper = p2
            self.iv_lower = p1
            self.iv = (p2 << self.word_size) + p1
            
        elif self.mode == "ECB":
            p2, p1 = self.encrypting(p2, p1)
            
        ct = (p2 << self.word_size) + p1
        
        return ct
    
########################################################################################################################################################################
###DECRYPTION
    def decryptround(self, p1,p2,k):
        """Complete One Round of Inverse Feistel Operation"""

        xor_p1p2 = p1 ^ p2

        y = ((xor_p1p2 << (self.word_size - self.beta_shift)) + (xor_p1p2 >> self.beta_shift)) & self.mod_mask

        xor_p1k = p1 ^ k

        msub = ((xor_p1k - y) + self.mod_mask_sub) % self.mod_mask_sub

        x = ((msub >> (self.word_size - self.alpha_shift)) + (msub << self.alpha_shift)) & self.mod_mask

        return x, y
        
    def decrypting(self,p1,p2):
       
       x = p1
       y = p2

       # Run Encryption Steps For Appropriate Number of Rounds
       for k in reversed(self.key_schedule): 
           xor_xy = x ^ y

           y = ((xor_xy << (self.word_size - self.beta_shift)) + (xor_xy >> self.beta_shift)) & self.mod_mask

           xor_xk = x ^ k 

           msub = ((xor_xk - y) + self.mod_mask_sub) % self.mod_mask_sub 

           x = ((msub >> (self.word_size - self.alpha_shift)) + (msub << self.alpha_shift)) & self.mod_mask

           
       return x,y
   
    def decrypt(self,ciphertext):
        p2 = (ciphertext >> self.word_size) & self.mod_mask
        p1 = ciphertext & self.mod_mask
        
        if self.mode == 'CBC':
            f, e = p2, p1
            p2, p1 = self.decrypting(p2, p1)
            p2 ^= self.iv_upper
            p1 ^= self.iv_lower

            self.iv_upper = f
            self.iv_lower = e
            self.iv = (f << self.word_size) + e
            
        elif self.mode == 'ECB':
            p2, p1 = self.decrypting(p2, p1)
            
        pt = (p2 << self.word_size) + p1
        
        return pt
            
# cipher = SpeckCipher(24567, 128, 128, 'ECB')
# t = cipher.encrypt(65535)
# print(hex(t))

# z = cipher.decrypt(t)
# print(hex(z))            
            
            
            
            
            
            
            
            
            
            
         