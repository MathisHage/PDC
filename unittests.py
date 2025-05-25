import unittest
import project as pj
import random
import numpy as np
import math

class Tests(unittest.TestCase):
    
    def channel(self, x):
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
    
    def get_seq(self):
        m = ""
        for i in range(40):
            x = random.randint(0, 63)
            m += pj.index_to_char(x)
        return m
    
    def test_char_index(self):
        m = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ."
        
        i = 0
        
        for c in m:
            self.assertEqual(pj.char_index(c), i)
            i += 1
            
        self.assertEqual(pj.char_index('?'), -1)
    
    def test_index_to_char(self):
        m = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ."
        
        for i in range(64):
            self.assertEqual(pj.index_to_char(i), m[i])
            
        self.assertEqual(pj.index_to_char(-1), '?')
    
    def test_int_to_binary(self):
        i = 10
        value = pj.int_to_binary(np.array([i, i+1]))
        expected= np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1])
        
        for j in range(value.shape[0]):
            self.assertEqual(expected[j], value[j])
        
    def test_encoder(self):
        
        m = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"
        
        X = pj.encoder(m, 5, 2000/(40*6))
        
        Energy = np.linalg.norm(X)**2
        
        
    
    def test_end_to_end_1(self):
        m = self.get_seq()
        
        eps_b = 1000/(40*6)
        
        X = pj.encoder(m, 11, eps_b) 
        print(f"E = {np.linalg.norm(X)**2}")
        
        R = self.channel(X)
        
        eta = 0.44
        
        m_dec = pj.decoder(R, 11, 10)
        
        self.assertEqual(m, m_dec)
        
    
    def test_error_prob(self):
        
        res = np.empty((7), dtype=float)
        
        loop = 100
        
        wanted_energy = 1999
        
        for r in range(5, 12):
            print(f"Begin error for r = {r}")
            eps_b = wanted_energy/(math.ceil(240/(r+1))*(r+1))
            err = 0
            for i in range(loop):
                m = self.get_seq()
                print(f"Test #{i}")
            
                X = pj.encoder(m, r, eps_b)  
                print(f"E = {np.linalg.norm(X)**2}")
                
                R = self.channel(X)
                
                m_dec = pj.decoder(R, r, 10)
                
                if m_dec != m:
                    err += 1
                    print(f"   Error ! err = {err}, X = {m_dec}")
            res[r-5] = err*100/loop
            print(f"End error for r = {r}. Final val = {err}. Error prob = {err*100/loop}%")
            
        print(res)
        
        

if __name__ == '__main__':
    unittest.main()