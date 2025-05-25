#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 18:27:58 2025

@author: hage

The docstring documentation of some functions may have been created 
using AI tools such as ChatGPT or Github Copilot.
"""


import numpy as np
import argparse
import random
import math

#------------- UTF-8 constants --------------
LOWERCASE_OFFSET = 0
LOWERCASE_START_IDX = 0x61
LOWERCASE_END_IDX = 0x7a

UPPERCASE_OFFSET = LOWERCASE_OFFSET + 26
UPPERCASE_START_IDX = 0x41
UPPERCASE_END_IDX = 0x5a

DIGIT_OFFSET = UPPERCASE_OFFSET + 26
DIGIT_START_IDX = 0x30
DIGIT_END_IDX = 0x39

SPACE_OFFSET = DIGIT_OFFSET + 10

POINT_OFFSET = SPACE_OFFSET + 1
#--------------------------------------------

LEGAL_CHARS_SEQ = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ."
CHAR_NUMBER = len(LEGAL_CHARS_SEQ)

MIN_R_VALUE = math.floor(math.log2(CHAR_NUMBER)) - 1

SEQUENCE_LENGTH = 40

def m_r(r: int):
    if r <= 0:
        return np.ones((1, 1))
    
    try:
        arr = np.load(f"res/matrix/m_{r}")
    except FileNotFoundError:
        array = m_r(r-1)
        
        top = np.concatenate((array, array), axis = 1)
        bottom = np.concatenate((array, -array), axis = 1)
        
        arr = np.concatenate((top, bottom), axis = 0)
        
        np.save(f"res/matrix/m_{r}", arr)
        
    return arr
    

def b_r(r: int):
    M = m_r(r)
    
    top = np.concatenate((M, M), axis = 1)
    bottom = np.concatenate((-M, -M), axis = 1)
    
    return np.concatenate((top, bottom), axis = 0)

def char_index(c: str):
    assert(len(c) == 1)
    
    code = ord(c)
    
    if LOWERCASE_START_IDX <= code and code <= LOWERCASE_END_IDX:   # Lowercase letter.
        code -= LOWERCASE_START_IDX - LOWERCASE_OFFSET
        
    elif UPPERCASE_START_IDX <= code and code <= UPPERCASE_END_IDX: # Uppercase letter.
        code -= UPPERCASE_START_IDX - UPPERCASE_OFFSET
        
    elif DIGIT_START_IDX <= code and code <= DIGIT_END_IDX:         # Digit.
        code -= DIGIT_START_IDX - DIGIT_OFFSET
        
    elif c == ' ':
        code = SPACE_OFFSET
    
    elif c == '.':
        code = POINT_OFFSET
        
    else:
        return -1
        
    return code


def index_to_char(idx: int):
    if 0 <= idx <= CHAR_NUMBER:
        return LEGAL_CHARS_SEQ[idx]
    else:
        return '?'
    
    
    
def int_to_binary(x, nb_bits = MIN_R_VALUE+1):
    return np.array((((x[:,None] & (1 << np.arange(nb_bits))[::-1])) > 0).astype(int)).flatten()

def binary_to_int(bits: np.ndarray) -> int:
    return int(bits.dot(1 << np.arange(bits.size)[::-1]))
    

def encoder(message: str, r: int, epsilon: float):
    """
    Encodes a fixed-length string into a vector to be sent on the noisy
    channel.

    Parameters
    ----------
    message : str
        Input string of length SEQUENCE_LENGTH containing letters, digits, space, or period.
    r : int
        B_r matrix parameter (must have r >= MIN_R_VALUE).
    epsilon : float
        Desired energy per bit.

    Returns
    -------
    np.ndarray
        Encoded 1D array of shape (SEQUENCE_LENGTH * 2**r,).

    Raises
    ------
    AssertionError
        If input constraints are not met.
    """
    
    assert(len(message) == SEQUENCE_LENGTH)
    assert(r >= MIN_R_VALUE)    # Otherwise we cannot encode each of the 64 characters.
    
    B = b_r(r)
    
    
    # Will contain int values of the characters in the message.
    m_int = np.empty((SEQUENCE_LENGTH), dtype = int)
    # Encode each character and put it in X
    idx = 0   
    for c in message:
        m_int[idx] = char_index(c)
        idx += 1
        
    # Convert in a contiguous binary array.
    m_bin = int_to_binary(m_int)
    
    # To have an integer number of c_i's.
    if m_bin.shape[0] % (r+1) != 0:
        m_bin = np.append(m_bin, np.zeros(((r+1) - (m_bin.shape[0] % (r+1)))))
        
    X = np.empty(shape = (m_bin.shape[0]//(r+1), B.shape[1]), dtype = int)
    
    
    idx = 0
    for i in range(X.shape[0]):
        X[i] = B[binary_to_int(m_bin[idx:(idx+r+1)])]
        
        # Interleave : put the first half of the elements as the even components, 
        # and the second half as the odd components
        temp1 = X[i][0:2**r]
        temp2 = X[i][2**r:]
        out = np.empty(shape = (2**(r+1)))
        out[::2] = temp1
        out[1::2] = temp2
        X[i] = out
        
        idx += r+1
        
    
    alpha = math.sqrt(epsilon*(r+1)/(2**(r+1)))
    return X.flatten('C')*alpha

def decoder(x: np.ndarray, r: int, G: float = 10):
    
    assert(r >= MIN_R_VALUE)    # Otherwise we cannot encode each of the 64 characters.
    
    n = x.shape[0]
    
    # The number of c_i's contained in X.
    nb_c = math.ceil(40 * 6 / (r+1))
    
    # The length of a c_i.
    length_c = 2**(r+1)
    
    B = b_r(r)
    
    # De-interleave the codewords, i.e. put the even coefs as the top half, and
    # the odd coefs as the bottom half of each c_i.
    for i in range(0, x.shape[0], 2**(r+1)):
        temp_even = x[i:(i+length_c):2]
        temp_odd = x[i+1:(i+length_c):2]
        
        out = np.empty(length_c)
        
        out[0:length_c//2] = temp_even
        out[length_c//2:length_c] = temp_odd
        
        x[i:i+length_c] = out
    
    # s = 1 : G occurs in the even components
    x_1 = x.copy()
    # s = 2 : G occurs in the odd components
    x_2 = x.copy()


    # Multiply each top (x_1) / bottom (x_2) coefs by sqrt(G), to implement the
    # decoding rule.
    for i in range(0, n, length_c):
        x_1[i:i+length_c//2] *= math.sqrt(G)
        x_2[i+length_c//2:i+length_c] *= math.sqrt(G)
    
    
    # Matrix with the columns being every Y
    x_1_mat = x_1.reshape(nb_c, length_c).T
    x_2_mat = x_2.reshape(nb_c, length_c).T
    
    # Matrix with (i, j) coefficient being the scalar product of c_i with the
    # j^th Y (with the factor sqrt(G) added before).
    score_1 = B @ x_1_mat
    score_2 = B @ x_2_mat
        
    
    
    # compute score(i, Y) for each elem
    score = np.maximum(score_1, score_2)
    
    # Max value indexes of the scores.
    argmax = np.argmax(score, axis=0)
    
    bit_tab = int_to_binary(argmax, r+1)
    
    m = ""
    # Retrieve the characters (consecutive sequences of 6 bits in `bit_tab`) and
    # get the corresponding symbol.
    for i in range(0, SEQUENCE_LENGTH * (MIN_R_VALUE + 1), MIN_R_VALUE + 1):
        m += index_to_char(binary_to_int(bit_tab[i:(i+MIN_R_VALUE+1)]))
        
    return m
        
            
    
    
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, help="The value of r to compute B_r (default = 2)", default=5)
    args = parser.parse_args()
        
    X = encoder("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN", args.r, 2**(args.r+1)/(args.r+1))
    #print(f"Energy : {np.linalg.norm(X)**2:.2f} J, shape : {X.shape}")
    
    m = decoder(X, args.r, 1)
    print(f"Message: {m}")
    

if __name__ == "__main__":
    main()
