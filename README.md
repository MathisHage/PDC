# PDC project : simple encoder

## Choice of implementation

### Encoder

The idea behind the encoder is the following. We have a sequence of 40 characters out of 64 possible symbols, which are hence treated as a $40 \times 6 = 240$ bits of data. The user chooses the number of bits to encode in a single codeword. This number of bits is defined in the code as $r+1$. Then, the codewords for each set of $r+1$ bits is chosen from the rows of $\sqrt{\alpha} [B_r  B_r]$ as seen in the theory part : the integer value (unsigned) of these bits serves to choose the row. Each of the rows chosen are then "interleaved" meaning the top half becomes the even components, and the bottom half becomes he odd components, to pass through the channel. Finally, all the codewords are concatenated in a single vector $X$ which is outputed by the encoder.

### Decoder

To decode the vector $R$ going out of the channel, the decoder implements the decoding rule $\hat{i}_1$ seen in exercise two of the theory part. First, the interleaving operation done in the encoder is undone. Then, a separation is made based on the two possible states that the channel is able to take : $R$ is copied with the factor $\sqrt{G}$ multiplied with the even components, and copied another time with $\sqrt{G}$ multiplied with the odd components. The score are then computed via matrix products. Finally, the guess of the integer values made by the decoder are converted to binary (because they were encoded with $r+1$ bits), and then converted back to integer considering sequences of 6 bits. The corresponding symbols are lastly outputted.
