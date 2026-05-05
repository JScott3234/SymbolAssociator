import drawingWindow as dw
import numpy as np
from brian2 import *
import os

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights.npy")
LETTERS = ['A', 'B', 'C'] # TODO: update patternSet templates when I expand it

# Pattern Templates for letters
# Some of these typographies are not as detailed, ones with less detail should be queried first so that they have preference over 
# more detailed letters to avoid ignoring the less striking ones.
def patternSet():
    # 10x10 - set in drawingWindow file

    # Templates are drawn with array splices, using simple uppercase letters
    # !NOTE! everything is rotated -90 in the process of compressing
    A = np.zeros((dw.gridSize, dw.gridSize))
    A[1:9, 2] = 1; A[1:9, 7] = 1
    A[1, 3:7] = 1; A[5, 3:7] = 1

    B = np.zeros((dw.gridSize, dw.gridSize))
    B[1:9, 2] = 1; B[1, 2:7] = 1; B[4, 2:7] = 1; B[8, 2:7] = 1
    B[2:4, 7] = 1; B[5:8, 7] = 1
    
    C = np.zeros((dw.gridSize, dw.gridSize))
    C[1:9, 2] = 1; C[1, 3:8] = 1; C[8, 3:8] = 1; C[2,8] = 1; C[7,8] = 1
    
    # Reduced each to 1D arrays
    return [A.flatten(), B.flatten(), C.flatten()]

def getWeights(N_input, N_output):
    if os.path.exists(WEIGHTS_PATH):
        return np.load(WEIGHTS_PATH)
    
    weights = np.zeros((N_input, N_output))
    patterns = patternSet()
    for j in range(N_output):
        pattern = patterns[j]
        numPixels = np.sum(pattern)
        num_zeros = N_input - numPixels
        excitatoryWeight = (120 / numPixels)
        inhibitoryWeight = (-100 / num_zeros)
        for i in range(N_input):
            if pattern[i] == 1:
                weights[i, j] = excitatoryWeight
            else:
                weights[i, j] = inhibitoryWeight
                
    np.save(WEIGHTS_PATH, weights)
    return weights

def letterIdentification(grid_data, train_target=None):
    
    start_scope()
    
    # I Declare 4 Quintillion Inputs, it's got what plants crave
    N_input = len(grid_data) # grid_data is just gridSize constant squared and flattened for processing (see drawingWindow)
    rates = grid_data * 150 * Hz # inputs fire at 150Hz
    P_in = PoissonGroup(N_input, rates=rates) # Poisson Spikes for stochastic behavior
    
    # Outputs for each letter ('cause they're attributed exclusively to each)
    N_output = len(LETTERS) # I made this flexible in case I expand to more letters
    
    # Izhikevich model equations
    eqs = '''
    dv/dt = (0.04/ms/mV)*v**2 + (5/ms)*v + 140*mV/ms - u/ms + I/ms : volt
    du/dt = a*(b*v - u) : volt
    I : volt
    a : Hz
    b : 1
    c : volt
    d : volt
    '''
    
    # Threshold and reset conditions for Izhikevich
    gOut = NeuronGroup(N_output, eqs, threshold='v >= 30*mV', 
                        reset='v = c; u += d', method='euler')
    
    # Regular Spiking Parameters
    gOut.a = 0.02 * Hz
    gOut.b = 0.2
    gOut.c = -65 * mV
    gOut.d = 8 * mV
    gOut.v = -65 * mV
    gOut.u = gOut.b * gOut.v
    gOut.I = 0 * mV
    
    # STDP parameters
    taupre = taupost = 20*ms
    Apre = 1.0 * mV
    Apost = -1.0 * mV
    wmax = 200 * mV
    wmin = -150 * mV

    eqs_stdp = '''
    w : volt
    dapre/dt = -apre/taupre : volt (event-driven)
    dapost/dt = -apost/taupost : volt (event-driven)
    '''
    on_pre = '''
    v_post += w
    apre += Apre
    w = clip(w + apost, wmin, wmax)
    '''
    on_post = '''
    apost += Apost
    w = clip(w + apre, wmin, wmax)
    '''
    
    # Synapses with STDP
    S = Synapses(P_in, gOut, eqs_stdp, on_pre=on_pre, on_post=on_post)
    S.connect()
    
    # Load weights from .npy
    saved_weights = getWeights(N_input, N_output)
    for j in range(N_output):
        for i in range(N_input):
            S.w[i, j] = saved_weights[i, j] * mV

    # Lateral Inhibition with i != j
    S_lat = Synapses(gOut, gOut, 'w_lat : volt', on_pre='v_post += w_lat')
    S_lat.connect(condition='i != j')

    # When one neuron spikes, it severely suppresses the others
    S_lat.w_lat = -50 * mV 

    # Monitor for spike stats and graph
    spikes = SpikeMonitor(gOut)
    
    # Training mode
    if train_target is not None:
        gOut.I[train_target] = 1000 * mV
        print(f"Training letter {LETTERS[train_target]}...")
    
    # Brian2 uses this in its compiler
    # i and j too, but the compiler resolves them
    if 'rates' in locals(): del rates
    
    # Runtime
    run(1000 * ms)
    
    # Save weights upon change
    if train_target is not None:
        new_weights = np.zeros((N_input, N_output))
        for j in range(N_output):
            for i in range(N_input):
                new_weights[i, j] = S.w[i, j] / mV
        np.save(WEIGHTS_PATH, new_weights)
        print("Weights saved to weights.npy")
    
    spikeCount = spikes.count
    
    maxSpikes = max(spikeCount)
    if maxSpikes == 0:
        return "None (No spikes)", None
    
    # returns letter with most spikes, always biased toward the first largest number of two in the sequence
    # What if I made it indecisive? Or maybe if it had a low enough spikage, it could not be sure of the letter?
    chosenLetter = np.argmax(spikeCount)

    result_text = f"{LETTERS[chosenLetter]} (Spikes: A={spikeCount[0]}, B={spikeCount[1]}, C={spikeCount[2]})"
    return result_text, spikes