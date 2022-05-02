# Author: Jacob Dawson
# This code is just my implementation of this walkthrough:
# https://analyticsindiamag.com/sequence-to-sequence-modeling-using-lstm-for-language-translation/
# I just wrote this up really quickly in order to teach myself keras!

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

batch_size = 64
epochs=100
latent_dim=256
num_samples=10000
data_path="fra.txt" # its right here in the same folder.

inputTexts = list()
targetTexts = list()
inputChars = set()
targetChars = set()
with open(data_path,'r') as f:
    lines = f.read().split("\n")
for line in lines[: min(num_samples,len(lines)-1)]:
    # not sure why this is formatted so weird???
    input_text, target_text, _ = line.split('\t')
    target_text = '\t' + target_text + '\t'
    inputTexts.append(input_text)
    targetTexts.append(target_text)
    for char in input_text:
        if char not in inputChars:
            inputChars.add(char)
    for char in target_text:
        if char not in targetChars:
            targetChars.add(char)

inputChars = sorted(list(inputChars))
targetChars = sorted(list(targetChars))
numEncoderTokens = len(inputChars)
numDecoderTokens = len(targetChars)
maxEncoderSeqLength = max([len(txt) for txt in inputTexts])
maxDecoderSeqLength = max([len(txt) for txt in targetTexts])

print("Number of samples:", len(inputTexts))
print("Number of unique input tokens:", numEncoderTokens)
print("Number of unique output tokens:", numDecoderTokens)
print("Max sequence length for inputs:", maxEncoderSeqLength)
print("Max sequence length for outputs:", maxDecoderSeqLength)

inputTokenIndex = dict([(char,i) for i, char in enumerate(inputChars)])
targetTokenIndex = dict([(char,i) for i, char in enumerate(targetChars)])

encoderInputData = np.zeros((len(inputTexts), maxEncoderSeqLength, numEncoderTokens), dtype = "float32")
decoderInputData = np.zeros((len(inputTexts), maxDecoderSeqLength, numDecoderTokens), dtype = "float32")
decoderTargetData = np.zeros((len(inputTexts), maxDecoderSeqLength, numDecoderTokens), dtype = "float32")

for i, (inputText, targetText) in enumerate(zip(inputTexts, targetTexts)):
    for t, char in enumerate(inputText):
        encoderInputData[i,t,inputTokenIndex[char]] = 1.0
    encoderInputData[i, t+1:, inputTokenIndex[" "]] = 1.0
    for t,char in enumerate(targetText):
        decoderInputData[i,t,targetTokenIndex[char]] = 1.0
        if t > 0:
            decoderTargetData[i, t-1, targetTokenIndex[char]] = 1.0
    decoderInputData[i, t+1 :, targetTokenIndex[" "]] = 1.0
    decoderTargetData[i, t:, targetTokenIndex[" "]] = 1.0

encoderInputs = keras.Input(shape=(None,numEncoderTokens))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoderOutputs, state_h, state_c = encoder(encoderInputs)

encoderStates = [state_h, state_c]

decoderInputs = keras.Input(shape=(None,numDecoderTokens))

decoderLSTM = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoderOutputs, _, _ = decoderLSTM(decoderInputs, initial_state = encoderStates)
decoderDense = keras.layers.Dense(numDecoderTokens, activation='softmax')
decoderOutputs = decoderDense(decoderOutputs)

model = keras.Model([encoderInputs, decoderInputs], decoderOutputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics='accuracy')
#model.fit([encoderInputData, decoderInputData], decoderTargetData, batch_size=batch_size, epochs=epochs, validation_split=0.2)

#model.save("s2s")
model = keras.models.load_model("s2s")

encoder_inputs = model.input[0] #input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1] #input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,),name='input_3')
decoder_state_input_c = keras.Input(shape=(latent_dim,),name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs=decoder_dense(decoder_outputs)

decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in inputTokenIndex.items())
reverse_target_char_index = dict((i, char) for char, i in targetTokenIndex.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, numDecoderTokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, targetTokenIndex["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > maxDecoderSeqLength:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, numDecoderTokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence

for seq_index in range(50):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    j = random.randrange(0, len(encoderInputData))
    input_seq = encoderInputData[j : j + 1]
    #print("looks like", encoderInputData[seq_index : seq_index + 1])
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", inputTexts[j])
    print("Decoded sentence:", decoded_sentence)

'''engCmd = ""
while(engCmd.lower() != 'exit'):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoderInputData[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", inputTexts[seq_index])
    print("Decoded sentence:", decoded_sentence)
'''
