import os
import sys

# Add utils path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)) + '/utils')

from ann_lib import AdaptiveSO2CPGSynPlas, Neuron

# ann = ANN(5)

# ann.setInput(4, 1)
# ann.setWeight(1, 2, 5)

# print(ann.dumpBiases())

# print(ann.dumpWeights())

M_PI=3.1415

neur = Neuron()

cpg = AdaptiveSO2CPGSynPlas(neur)

cpg.setPhi     ( 0.02*M_PI )   # Frequency term - Influences w00 w01 w10 w11 of the SO(2) oscillator (long term)
cpg.setEpsilon ( 0.1 )        # Value should depend on the initial and external freq - from P to h2 (short term)
cpg.setAlpha   ( 1.01)        # Amplitude and linearity between phi and the frequency
cpg.setGamma   ( 1.0 )        # Synaptic weight from h2 to h0 - Governed by a Hebbian-type learning (short term)
cpg.setBeta    ( 0.0 )        # Synaptic weight from h0 to h2 - Governed by a Hebbian-type learning (short term)
cpg.setMu      ( 1.0 )        # Learning rate - Value should depend on the given initial and external freq
cpg.setBetaDynamics( -1.0, 0.010, 0.00) # Heppian Rate, Decay Rate, Beta_0
cpg.setGammaDynamics( -1.0, 0.010, 1.00) # --- || ---
cpg.setEpsilonDynamics(  1.0, 0.010, 0.01) # --- || ---

#destabilize cpg to oscillate
cpg.setOutput(0,0.2012)
cpg.setOutput(1,0)

print(cpg.getGamma())