import os
import binascii
from binascii import unhexlify
from hashlib import pbkdf2_hmac
from hmac import new as new_hmac, compare_digest

#128 key length
s_box = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
]
inv_s_box = [
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
]
Rcon = [0x00000000, 0x01000000, 0x02000000,
		0x04000000, 0x08000000, 0x10000000, 
		0x20000000, 0x40000000, 0x80000000, 
		0x1b000000, 0x36000000]

def sub_bytes(s):
    for i in range (4):
        for j in range(4):
            s[i][j] = s_box[s[i][j]]
    return s
            
def inv_sub_bytes(s):
    for i in range(4):
        for j in range(4):
            s[i][j] = inv_s_box[s[i][j]]
    return s

#Shift rows:
#Encrypting
#1)not shifting the first row of the array
#2)shifting the second row by one byte to the left
#3)shifting the third row by two bytes to the left
#4)shifting the last row by three bytes to the left
def shift_rows(s):
    s[0][1], s[1][1], s[2][1], s[3][1] = s[1][1], s[2][1], s[3][1], s[0][1]
    s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    s[0][3], s[1][3], s[2][3], s[3][3] = s[3][3], s[0][3], s[1][3], s[2][3]
    
    return s

#Inverse Rows
#1)first row unchanged
#2)Second row is shifted to the right by one byte
#3)Third row to the right by two bytes
#4)Last row by three bytes
def inv_shift_rows(s):
    s[0][1], s[1][1], s[2][1], s[3][1] = s[3][1], s[0][1], s[1][1], s[2][1]
    s[0][2], s[1][2], s[2][2], s[3][2] = s[2][2], s[3][2], s[0][2], s[1][2]
    s[0][3], s[1][3], s[2][3], s[3][3] = s[1][3], s[2][3], s[3][3], s[0][3]
    
    return s

xtime = lambda a: (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)

#Mix single column:
#1) s′0,j = (0x02 × s0,j) ⊗ (0x03 × s1,j ) ⊗ s2,j ⊗ s3,j
#2) s′1,j = s0,j ⊗ (0x02 × s1,j) ⊗ (0x03 × s2,j) ⊗ s3,j
#3) s′2,j = s0,j ⊗ s1,j ⊗ (0x02 × s2,j ) ⊗ (0x03 × s3,j)
#4) s′3,j = (0x03 × s0,j) ⊗ s1,j ⊗ s2,j ⊗ (0x02 × s3,j)

def mix_single_column(a):
   
    t = a[0] ^ a[1] ^ a[2] ^ a[3]
    u = a[0]
    a[0] ^= t ^ xtime(a[0] ^ a[1])
    a[1] ^= t ^ xtime(a[1] ^ a[2])
    a[2] ^= t ^ xtime(a[2] ^ a[3])
    a[3] ^= t ^ xtime(a[3] ^ u)

def mix_columns(s):
    for i in range(4):
        mix_single_column(s[i])
    return s

def inv_mix_columns(s):
    # see Sec 4.1.3 in The Design of Rijndael
    for i in range(4):
        u = xtime(xtime(s[i][0] ^ s[i][2]))
        v = xtime(xtime(s[i][1] ^ s[i][3]))
        s[i][0] ^= u
        s[i][1] ^= v
        s[i][2] ^= u
        s[i][3] ^= v

    return mix_columns(s)
    
def add_round_key(s, k):
    for i in range(4):
        for j in range(4):
            s[i][j] ^= k[i][j]
    return s
            
def bytes2matrix(text):
    """ Converts a 16-byte array into a 4x4 matrix.  """
    return [list(text[i:i+4]) for i in range(0, len(text), 4)]

def matrix2bytes(matrix):
    """ Converts a 4x4 matrix into a 16-byte array.  """
    return bytes(sum(matrix, []))

def xor_bytes(a, b):
    """ Returns a new byte array with the elements xor'ed. """
    return bytes(i^j for i, j in zip(a, b))

#takes a hex value and returns binary
def hex2binary(hex):
    return bin(int(str(hex), 16))

def hexor(hex1, hex2):
    #convert to binary
    bin1 = hex2binary(hex1)
    bin2 = hex2binary(hex2)
    
    #calculate
    xord = int(bin1, 2) ^ int(bin2, 2)
    
    #cut prefix
    hexed = hex(xord)[2:]
    
    #leading 0s get cut above, if not length 8 add a leading 0
    if len(hexed) != 8:
        hexed = '0' + hexed
        
    return hexed

def Binary(a):
    l,m = [],[]
    for i in a:
        l.append(ord(i))
    for i in l:
        m.append(int(bin(i)[2:]))
    for i in m:
        list = [str(x).zfill(8) for x in m]
    for i in list:
           result = "".join(map(str,list))
    return result

def binToHexa(n):
    # convert binary to int
    num = int(n, 2)
      
    # convert int to hexadecimal
    hex_num = format(num, 'x')
    return(hex_num)

def xor(a, b):
    ans = ""
    for i in range(len(a)):
        if a[i] == b[i]:
            ans = ans + "0"
        else:
            ans = ans + "1"
    return ans

def pad(plaintext):
    #PKCS#7 padding to a multiple of 16 bytes.
    padding_len = 16 - (len(plaintext) % 16)
    padding = bytes([padding_len] * padding_len)
    
    return plaintext + padding

def unpad(plaintext):
   #PKCS#7 
    
    padding_len = plaintext[-1]
    assert padding_len > 0
    message, padding = plaintext[:-padding_len], plaintext[-padding_len:]
    assert all(p == padding_len for p in padding)
    return message

def split_blocks(message, block_size=16, require_padding=True):
        assert len(message) % block_size == 0 or not require_padding
        return [message[i:i+16] for i in range(0, len(message), block_size)]
    
#rotate bytes of the word    
def RotWord(word):
    return word[1:] + word[:1]

#selects correct value from sbox based on the current word
def SubWord(word):
    sWord = ()
    
    #loop throug the current word
    for i in range(4):
        
        #check first char, if its a letter(a-f) get corresponding decimal
        #otherwise just take the value and add 1
        if word[i][0].isdigit() == False:
            row = ord(word[i][0]) - 86
        else:
            row = int(word[i][0])+1

        #repeat above for the seoncd char
        if word[i][1].isdigit() == False:
            col = ord(word[i][1]) - 86
        else:
            col = int(word[i][1])+1
        
        #get the index base on row and col (16x16 grid)
        sBoxIndex = (row*16) - (17-col)
        
        #get the value from sbox without prefix
        piece = hex(s_box[sBoxIndex])[2:]
        
        #check length to ensure leading 0s are not forgotton
        if len(piece) != 2:
            piece = '0' + piece
        
        #form tuple
        sWord = (*sWord, piece)
        
    #return string
    return ''.join(sWord)

def array_tohexa(s):
    s = "".join(s)

    return s

def hexa_ascii(s):
    b = bytearray.fromhex(s).decode()
    return b

def hex_bytes(s):
    byte = unhexlify(s)
    
    return byte
def bytes_hexa(s):
    b = s.hex()
    
    return b

def key_hexalist(s):
    k = []
    c = bytes_hexa(s)
    
    for i in range (0,len(c),2):
        k.append(c[i:i+2])
        
    return k

def plaintext():
    pt = input("\nEnter Plaintext: ")
    pt = Binary(pt)
    pt = binToHexa(pt)
    pt = bytes.fromhex(pt)
    
    return pt

def bytes_key_array(key):
    key = key.hex()
    keys = []
    
    for i in range(0,len(key),2):
        keys.append(key[i:i+2])
        
    return keys

class AESCipher:
    
    def __init__(self, Key):
        self.key = self.key_expansion(Key)
        
    #key expansion function
    def key_expansion(self, key):
        if isinstance(key, bytes):
            key = bytes_key_array(key)   
        #list for 44 words
        w = [()]*44
        s = []
        d = []
        #1) slice key into 4 words
        for i in range(4):
            w[i] = (key[4*i], key[4*i+1], key[4*i+2], key[4*i+3])
            
        #2) Make the 4 words into 44 words by filling it out from previous words
        for i in range(4,44):
            temp = w[i-1]
            word = w[i-4]
            
            if i % 4 == 0:
                x = RotWord(temp)
                y = SubWord(x)
                rcon = Rcon[int(i/4)]
                temp = hexor(y, hex(rcon)[2:])
                
            word = ''.join(word)
            temp = ''.join(temp)
            
            xord = hexor(word,temp)
            w[i] = (xord[:2],xord[2:4],xord[4:6],xord[6:8])   
            
        for i in range (44):
            
            s.append(w[i])
        for i in range (44):
            alist = list(s[i])
            alist = [int(i,16) for i in alist]
            d.append(alist)
        
        return [d[4*i : 4*(i+1)] for i in range(len(d) // 4)]
    
    #Encryption
    def encrypt(self, plaintext):
        #pt is in bytes and converted to matrix
        pt = bytes2matrix(plaintext)
        #1key w0 - w3, add round key
        pt = add_round_key(pt, self.key[0])
        for i in range(1,4):
            pt = sub_bytes(pt)
            pt = shift_rows(pt)
            pt = mix_columns(pt)
            pt = add_round_key(pt, self.key[i])
    
        pt = sub_bytes(pt)
        pt = shift_rows(pt)
        pt = add_round_key(pt, self.key[-1])
        
        pt = matrix2bytes(pt)
        
        return pt 
    
    #Decryption
    def decrypt(self,ciphertext):
        #pt is in bytes and converted to matrix
        ct = bytes2matrix(ciphertext)
        
        #1key w0 - w3, add round key
        # ct = add_round_key(ct, self.key[-1])
        
        # ct = inv_shift_rows(ct)
        
        # ct = inv_sub_bytes(ct)
        
        for i in range(2 -1,0, -1):
            ct = add_round_key(ct, self.key[i])
            ct = inv_mix_columns(ct)
            ct = inv_shift_rows(ct)
            ct = inv_sub_bytes(ct)
            
            
        ct = add_round_key(ct, self.key[0])
        ct = matrix2bytes(ct)
        return ct
    
    #CBC MODE TO ENCRYPT PLAINTEXT
    def encrypt_cbc(self,plaintext,iv):
        
        assert len(iv) == 16
        
        plaintext = pad(plaintext)
        
        blocks = []
        previous = iv
        for plaintext_block in split_blocks(plaintext):
            # CBC mode encrypt: encrypt(plaintext_block XOR previous)
            block = self.encrypt(xor_bytes(plaintext_block, previous))
            blocks.append(block)
            previous = block

        return b''.join(blocks)

    def decrypt_cbc(self, ciphertext, iv):
        
        assert len(iv) == 16
        
        blocks = []
        previous = iv
        for ciphertext_block in split_blocks(ciphertext):
            # CBC mode decrypt: previous XOR decrypt(ciphertext)
            blocks.append(xor_bytes(previous, self.decrypt(ciphertext_block)))
            previous = ciphertext_block
        
        return unpad(b''.join(blocks))
    
#Getting Password from User
def get_key_from_user():
    key = input("\nEnter key (any number of chars): ")
    key = key.strip()
    keys = []
    #adding 0 to password that is less than 128 bits
    key += '0' * (128 // 8 - len(key)) if len(key) < 128 // 8 else key[:128 // 8]
    
    #converting key into binary
    key_bv = Binary(key)
    
    #converting binary to hexa
    key_bv = binToHexa(key_bv)

    for i in range (0,len(key_bv),2):
        keys.append(key_bv[i:i+2])
        
    #return hexa of user password (key)
    return (keys)

def key_iv(password, salt):
    
    key_stretch = pbkdf2_hmac('sha256', password, salt, 200000, 48)
    aes_key, key_stretch = key_stretch[:16], key_stretch[16:]
    hmac_key,key_stretch = key_stretch[:16], key_stretch[16:]
    iv = key_stretch[:16] 
    
    return aes_key,hmac_key,iv

def encrypting(key,plaintext):
    if isinstance(key, bytes):
        key = bytes_key_array(key) 
    key = array_tohexa(key)
    key = hexa_ascii(key)
    print(key)
    if isinstance(key, str):
        key = key.encode('utf-8')
    if isinstance(plaintext, str):
        plaintext = plaintext.encode('utf-8')   
    salt = os.urandom(16)
    key,hmac_key,iv = key_iv(key, salt)
    key = key_hexalist(key)
    ct = AESCipher(key).encrypt_cbc(plaintext,iv)
    hmac = new_hmac(hmac_key, salt + ct, 'sha256').digest()
    
    assert len(hmac) == 32
    
    return hmac + salt + ct

def decrypting(key,ciphertext):
    if isinstance(key, bytes):
        key = bytes_key_array(key) 
    key = array_tohexa(key)
    key = hexa_ascii(key)
    assert len(ciphertext) % 16 == 0
    
    assert len(ciphertext) >= 32
    
    if isinstance(key, str):
        key = key.encode('utf-8')
        
    hmac, ciphertext = ciphertext[:32], ciphertext[32:]
    salt, ciphertext = ciphertext[:16], ciphertext[16:]
    key, hmac_key, iv = key_iv(key, salt)
    
    ori_hmac = new_hmac(hmac_key, salt + ciphertext, 'sha256').digest()
    assert compare_digest(hmac, ori_hmac)
    key = key_hexalist(key)
    return AESCipher(key).decrypt_cbc(ciphertext, iv)

# pt = b'\x01' * 16
# # print(pt)
# a = get_key_from_user()

# b = AESCipher(a).encrypt(pt)
# print(b)

# # # d = b"1234567890"
# c = AESCipher(a).decrypt(b)
# print("ct: ",c)

# text = "5d41402abc4b2a76b9719d911017c592"
# result = bytes.fromhex(text)
# print(result)
# a = bytes.fromhex(text)
# k = key_expansion(get_key_from_user())

# e = encrypt(result, k)
# print(e)

# d = decrypt(e,k)
# print(d)

# if __name__ == '__main__':
#     text = "5d41402abc4b2a76b9719d911017c592"
#     result = bytes.fromhex(text)
#     b = get_key_from_user()
#     a = encrypting("hello", result)
#     print(a)
    