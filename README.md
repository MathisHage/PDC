# PDC project : simple encoder

## Choice of implementation

### Encoder

The idea behind the encoder is the following. We have a sequence of 40 characters out of 64 possible symbols, which are hence treated as a $40 \times 6 = 240$ bits of data. The user chooses the number of bits to encode in a single codeword. This number of bits is defined in the code as $r+1$. Then, the codewords for each set of $r+1$ bits is chosen from the rows of $\sqrt{\alpha} [B_r  B_r]$ as seen in the theory part : the integer value (unsigned) of these bits serves to choose the row. Each of the rows chosen are then "interleaved" meaning the top half becomes the even components, and the bottom half becomes he odd components, to pass through the channel. Finally, all the codewords are concatenated in a single vector $X$ which is outputed by the encoder.

### Decoder

To decode the vector $R$ going out of the channel, the decoder implements the decoding rule $\hat{i}_1$ seen in exercise two of the theory part. First, the interleaving operation done in the encoder is undone. Then, a separation is made based on the two possible states that the channel is able to take : $R$ is copied with the factor $\sqrt{G}$ multiplied with the even components, and copied another time with $\sqrt{G}$ multiplied with the odd components. The score are then computed via matrix products. Finally, the guess of the integer values made by the decoder are converted to binary (because they were encoded with $r+1$ bits), and then converted back to integer considering sequences of 6 bits. The corresponding symbols are lastly outputted.

## Using the program

The arguments are described in detail in the `help` section (`-h` argument). Here is an overview of the overall usage of the commands :
### Encoding :
Use the `e` / `encode` action (1st and only positional argument). If an input file is specified (`-i`), the message will be read from there. Otherwise, the sequence used is the one specified from the command line (`-seq`). If no sequence is specified, a pseudo-random one is generated. The sequence must be 40 characters long. You can also optionally specify the wanted energy of the output vecor (`-e`), the value of parameter $r$ (`-r`), and the output path (`-o`, default is `res/output.txt`).

### Decoding :
Use the `d` / `decode` action. You have to specify an input file (`-i`). Don't forget the value of $r$ (`-r`) if you did not use the default value when encoding.

#### Testing :
You can also encode / decode using a local test channel. Use the action `t` / `test`. You have to specify the parameters as if you were encoding normally. The resulting decoded message is displayed in the command line.

## Results

The best efficient value of $r$ that we used is the value $r = 11$ (which is the default value for $r$). In fact, it corresponds to assigning exactly one codeword per two characters (so now redundant padding bits are sent), and the size of $\sqrt{\alpha} [B_r  B_r]$ allows for reasonably quick calculations. Here are a few results
on the error probability achived :

### 1. Error probability as a function of energy ($||X||^2$), if an "error" is an incorrectly decoded message :
<p align="center"><img width="450" alt="err_prob_per_seq" src="https://github.com/user-attachments/assets/02ae0597-579c-40ff-a900-45d0a3467620" /></p>

### 2. Error probability as a function of energy ($||X||^2$), if an "error" is an incorrectly decoded symbol :
<p align="center"><img width="450" alt="err_prob_per_symb" src="https://github.com/user-attachments/assets/2311c3ac-c587-457d-b1a0-37912be0601e" /></p>


*Note : these graphs were obtained by testing 300 times the encoding / decoding of a pseudo-randomly generated sequence on the local test channel (with r = 11), and this for values of the energy between 1000 and 1999, with a step of 100.*
