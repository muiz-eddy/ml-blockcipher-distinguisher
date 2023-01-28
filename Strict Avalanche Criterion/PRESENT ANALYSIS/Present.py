# coding: utf-8
from __future__ import print_function
from cgi import test

s_box = (0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2)

inv_s_box = (0x5, 0xE, 0xF, 0x8, 0xC, 0x1, 0x2, 0xD, 0xB, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xA)

p_layer_order = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38,
                 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13,
                 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63]

block_size = 64

ROUND_LIMIT = 32


def round_function(state, key):
    new_state = state ^ key
    state_nibs = []
    for x in range(0, block_size, 4):
        nib = (new_state >> x) & 0xF
        sb_nib = s_box[nib]
        state_nibs.append(sb_nib)
    # print(state_nibs)

    state_bits = []
    for y in state_nibs:
        nib_bits = [1 if t == '1'else 0 for t in format(y, '04b')[::-1]]
        state_bits += nib_bits
    # print(state_bits)
    # print(len(state_bits))

    state_p_layer = [0 for _ in range(64)]
    for p_index, std_bits in enumerate(state_bits):
        state_p_layer[p_layer_order[p_index]] = std_bits

    # print(len(state_p_layer), state_p_layer)

    round_output = 0
    for index, ind_bit in enumerate(state_p_layer):
        round_output += (ind_bit << index)

    # print(format(round_output, '#016X'))

    # print('')
    return round_output


def key_function_80(key, round_count):
    # print('Start: ', hex(key))
    # print('')

    r = [1 if t == '1'else 0 for t in format(key, '080b')[::-1]]

    # print('k bits:', r)
    # print('')

    h = r[-61:] + r[:-61]

    # print('s bits:', h)
    # print('')

    round_key_int = 0
    # print('init round int:', hex(round_key_int))
    for index, ind_bit in enumerate(h):
        round_key_int += (ind_bit << index)
        # print('round:',index, '-', hex(round_key_int))

    # print('round_key_int', hex(round_key_int))
    # print('')

    upper_nibble = round_key_int >> 76

    # print('upper_nibble:', upper_nibble)

    upper_nibble = s_box[upper_nibble]

    # print('upper_nibble sboxed', hex(upper_nibble))

    xor_portion = ((round_key_int >> 15) & 0x1F) ^ round_count
    # print('Count:', round_count)
    # print('XOR Value:', xor_portion)

    # print('Before:', hex(round_key_int))
    round_key_int = (round_key_int & 0x0FFFFFFFFFFFFFF07FFF) + (upper_nibble << 76) + (xor_portion << 15)
    # print('After: ', hex(round_key_int))

    return round_key_int


def key_function_128(key, round_count):
    # print('Start: ', hex(key))
    # print('')

    r = [1 if t == '1'else 0 for t in format(key, '0128b')[::-1]]

    # print('k bits:', r)
    # print('')

    h = r[-61:] + r[:-61]

    # print('s bits:', h)
    # print('')

    round_key_int = 0
    # print('init round int:', hex(round_key_int))
    for index, ind_bit in enumerate(h):
        round_key_int += (ind_bit << index)
        # print('round:',index, '-', hex(round_key_int))

    # print('round_key_int', hex(round_key_int))
    # print('')

    upper_nibble = (round_key_int >> 124) & 0xF
    second_nibble = (round_key_int >> 120) & 0xF
    # print('upper_nibble:', upper_nibble)

    upper_nibble = s_box[upper_nibble]
    second_nibble = s_box[second_nibble]

    # print('upper_nibble sboxed', hex(upper_nibble))

    xor_portion = ((round_key_int >> 62) & 0x1F) ^ round_count
    # print('Count:', round_count)
    # print('XOR Value:', xor_portion)

    # print('Before:', hex(round_key_int))
    round_key_int = (round_key_int & 0x00FFFFFFFFFFFFF83FFFFFFFFFFFFFFF) + (upper_nibble << 124) + (second_nibble << 120) + (xor_portion << 62)
    # print('After: ', hex(round_key_int))

    return round_key_int



def encryption(pt, key, rounds):
    key_schedules = []
    
    for i in range(rounds):
        key_schedules.append(key >> 16)
        key = key_function_80(key, i + 1)
    
    for i in range(rounds - 1):
        pt = round_function(pt, key_schedules[i])
    
    pt ^= key_schedules[rounds - 1]
    
    return pt


# test = encryption(0x0000000000000000, 0x00000000000000000000, 32)

# if test == 0x5579C1387B228445:
#     print('Success', hex(test))
    
# else:
#     print('Failure', hex(test))
    

# key_schedules = []
# current_round_key = 0x00000000000000000000
# round_state = 0x0000000000000000


# for i in range(ROUND_LIMIT):
#     key_schedules.append(current_round_key >> 16)
#     current_round_key = key_function_80(current_round_key, i + 1)
    
# for rnd in range(ROUND_LIMIT - 1):
#     # print('Round:', rnd)
#     # print('State:', format(round_state, '016X'))
#     # print('R_Key:', format(key_schedule[rnd], '016X'))
#     round_state = round_function(round_state, key_schedules[rnd])
    
# round_state ^= key_schedules[ROUND_LIMIT-1]

# print("")

# if round_state == 0x5579C1387B228445:
#     print('Success', hex(round_state))
    
# else:
#     print('Failure', hex(round_state))
    
    
    
    
    
    


