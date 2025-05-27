#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 18:27:58 2025

The docstring documentation of some functions may have been created 
using AI tools such as ChatGPT or Github Copilot.
"""


import numpy as np
import argparse
import pathlib
import random
import math
import sys
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

G = 10

ARG_ENCODE = ["e", "encode"]
ARG_DECODE = ["d", "decode"]
ARG_TEST = ['t', 'test']
ARG_FULL_FLOW = ['ff', 'full_flow']

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

def random_char_seq():
    m = ""
    for i in range(SEQUENCE_LENGTH):
        x = random.randint(0, CHAR_NUMBER-1)
        m += index_to_char(x)
    return m
    
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

def decoder(x: np.ndarray, r: int, G: float = G):
    
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

def handle_args():
    parser = argparse.ArgumentParser(description="Encode or decode messages using the custom encoding scheme.")
    parser.add_argument("action", 
                        choices = ARG_ENCODE + ARG_DECODE + ARG_TEST + ARG_FULL_FLOW,
                        help="'e'/'encode' to encode, 'd'/'decode' to decode, 't'/'test' to test encoding and decoding on the local test channel, 'ff' / 'full_flow' to encode, send to the server and decode (must be on the EPFL network).")
    parser.add_argument("-r", "--r" ,
                        type=int, 
                        help="The value of r to compute B_r (default = 11).", 
                        default=11)
    parser.add_argument("-e", "--energy" ,
                        type=float, 
                        help="The desired energy of the vector (default = 1800).", 
                        default=1800)
    parser.add_argument("-seq", "--sequence",
                        type=str, 
                        help=f"The {SEQUENCE_LENGTH}-symbols sequence to use (default is pseudo-randomly generated). If an input file (--input) is specified, this parameter will be disregarded.",
                        default=random_char_seq())
    parser.add_argument("-i", "--input",
                        help="The input file to encode / decode from.",
                        default="")
    parser.add_argument("-o", "--output",
                        help="Where to write the encoded data.",
                        default="res/output.txt")
    
    return parser.parse_args()

def test_channel(x):
    G = 10
    sigma2 = 10
    s = random.choice([1, 2])
    n = x.size
    Y = np.random.normal(0, np.sqrt(sigma2), n)
    if s == 1:
        x_even = np.array(x[::2]) * np.sqrt(G)
        x_odd = x[1::2]
    else:
        x_even = np.array(x[::2])
        x_odd = x[1::2] * np.sqrt(G)
    Y[::2] += x_even
    Y[1::2] += x_odd
    return Y

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def read_input_sequence(filename: str, default_val: str):
    if filename != "":
        try:
            file = open(filename, 'r')
            m = file.read().strip()
            file.close()
            if len(m) != SEQUENCE_LENGTH:
                eprint(f"The number of characters in the file '{filename}' is not {SEQUENCE_LENGTH} !")
                return -1
            return m
        except FileNotFoundError:
            print(f"File '{filename}' not found, will encode the value passed as argument (-seq).")
            
    if len(default_val) != SEQUENCE_LENGTH:
        eprint(f"The number of characters of the sequence passed as argument is not {SEQUENCE_LENGTH} !")
        return -1
    
    return default_val

def encode(m: str, r: int, e_b: float, output_file: str):
        
    print(f"Encoding the sequence '{m}' ...")
    X = encoder(m, r, e_b)
    print(f"||X||^2 = {round(np.linalg.norm(X)**2):.2f}, n = {X.shape[0]}")
    
    np.savetxt(output_file, X)
    
    print(f"Data saved in {output_file}")
    
def decode(input_file: str, r: int):
    if input_file == "":
        eprint("No specified input file ! (use -i [path]).")
        return -1
    
    # Load file
    try:
        R = np.loadtxt(input_file)
    except FileNotFoundError:
        eprint(f"The file {input} was not found.")
        return -1
    
    print("Decoding...")
    m = decoder(R, r)
    
    print(f"The decoded message is : '{m}'.")
    return m
    

def main():
    args = handle_args()
    os.makedirs("res/", exist_ok=True)
    
    # energy per bit
    e_b = args.energy / (math.ceil(SEQUENCE_LENGTH * (MIN_R_VALUE+1)/(args.r+1))*(args.r+1))
    
    if args.action in ARG_ENCODE:
        # Choose the message to encode based on the arguments (priority given
        # to the input file, if specified).
        m = read_input_sequence(args.input, args.sequence)
        if m == -1:
            return -1
        
        encode(m, args.r, e_b, args.output)
    
    elif args.action in ARG_DECODE:
        decode(args.input, args.r)

    elif args.action in ARG_TEST: # Test case
        # Choose the message to encode based on the arguments (priority given
        # to the input file, if specified).
        m = read_input_sequence(args.input, args.sequence)
        if m == -1:
            return
        
        print(f"Encoding the sequence '{m}' ...")
        X = encoder(m, args.r, e_b)
        print(f"||X||^2 = {round(np.linalg.norm(X)**2):.2f}, n = {X.shape[0]}")
        
        print("Applying the channel effects...")
        R = test_channel(X)
        
        print("Decoding...")
        m_dec = decoder(R, args.r)
        
        print(f"The decoded message is '{m_dec}'.")
        
        diff = sum ( m_dec[i] != m[i] for i in range(SEQUENCE_LENGTH) )
        print(f"The decoder failed in {diff} positions.")
        
    else: # Full flow case
    
        Rcv_file = "res/receive.txt"
        
        m = read_input_sequence(args.input, args.sequence)
        if m == -1:
            return -1
    
        encode(m, args.r, e_b, args.output)
        
        # Send the data to the server.
        print("Sending the data to the server...")
        
        res = os.system(f"python3 client/client.py --input_file {args.output} --output_file {Rcv_file} --srv_hostname=iscsrv72.epfl.ch --srv_port=80")
            
        if res >> 8 != 0:
            eprint("Error during the communication with the server. Exiting.")
            return
        
        print("Noisy vector received.")
        
        m_dec = decode(Rcv_file , args.r)
        
        # Delete the file where the data from the server was received.
        pathlib.Path.unlink(Rcv_file)
        
        if m_dec == -1:
            return
        
        diff = sum ( m_dec[i] != m[i] for i in range(SEQUENCE_LENGTH) )
        print(f"The decoder failed in {diff} positions.")
    

if __name__ == "__main__":
    main()
