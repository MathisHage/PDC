#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 18:27:58 2025

The docstring documentation of some functions may have been created 
using AI tools such as ChatGPT or Github Copilot.
"""


import numpy as np
import argparse
import random
import math
import os

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
        return np.ones((1, 1), dtype=int)
    
    try:
        arr = np.load(f"res/matrix/m_{r}.npy")
    except FileNotFoundError:
        array = m_r(r-1)
        
        top = np.concatenate((array, array), axis = 1)
        bottom = np.concatenate((array, -array), axis = 1)
        
        arr = np.concatenate((top, bottom), axis = 0)
        
        os.makedirs("res/matrix", exist_ok=True)
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
        
    # Convert to a contiguous binary array.
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
        
def read_text_file(file_path: str) -> str:
   #read the file and return the contents"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    print("the message is " + len(content) + " character long" )
    
    return content

def write_text_file(file_path: str, content: str) -> None:
   #write the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def encode_file(input_file: str, output_file: str, r: int, epsilon: float) -> None:
    message = read_text_file(input_file)
    encoded_data = encoder(message, r, epsilon)
    encoded_str = ' '.join(map(str, encoded_data))
    write_text_file(output_file, encoded_str)

def decode_file(input_file: str, output_file: str, r: int, G: float = 10) -> None:
    with open(input_file, 'r', encoding='utf-8') as f:
        encoded_str = f.read().strip()

    encoded_data = np.array([float(x) for x in encoded_str.split()])
    decoded_text = decoder(encoded_data, r, G)
    
    write_text_file(output_file, decoded_text)


def main():
    parser = argparse.ArgumentParser(description='Encode or decode messages using the custom encoding scheme.')
    parser.add_argument("--r", type=int, help="The value of r to compute B_r (default = 5)", default=5)
    parser.add_argument("--mode", choices=['encode', 'decode'], required=True, 
                       help="Operation mode: encode or decode")
    parser.add_argument("--message", help="Message to encode (default: random sequence)")
    parser.add_argument("--message-file", help="File containing message to encode")
    parser.add_argument("--energy", type=float, default=1900.0,
                       help="Total energy for encoding (default = 1900.0)")
    parser.add_argument("--input", help="Input file name (default: encoded.txt for decode mode)")
    parser.add_argument("--output", help="Output file name (default: encoded.txt for encode mode)")
    
    args = parser.parse_args()
    
    # Calculate epsilon based on total energy
    epsilon = args.energy / (SEQUENCE_LENGTH * (args.r + 1))
    
    if args.mode == 'encode':
        if args.message_file:
            try:
                message = read_text_file(args.message_file)
                if len(message) != SEQUENCE_LENGTH:
                    print(f"Error: Message must be exactly {SEQUENCE_LENGTH} characters long")
                    return 1
            except FileNotFoundError:
                print(f"Error: Message file {args.message_file} not found")
                return 1
        else:
            message = args.message if args.message else generate_random_sequence()
            
        encoded_data = encoder(message, args.r, epsilon)
        # Save encoded data to file
        output_file = args.output if args.output else 'encoded.txt'
        np.savetxt(output_file, encoded_data)
        print(f"Encoded message: {message}")
        print(f"Encoded data saved to {output_file}")
        
    else:  # decode mode
        input_file = args.input if args.input else 'encoded.txt'
        try:
            encoded_data = np.loadtxt(input_file)
            decoded_text = decoder(encoded_data, args.r)
            output_file = args.output if args.output else 'decoded.txt'
            write_text_file(output_file, decoded_text)
            print(f"Decoded message: {decoded_text}")
            print(f"Decoded text saved to {output_file}")
        except FileNotFoundError:
            print(f"Error: {input_file} not found. Please encode a message first.")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
