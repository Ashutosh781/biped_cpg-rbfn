#!/usr/bin/env python3

import os
import sys
import json
import random as rand

# Add project root to the python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.ann_lib import AdaptiveSO2CPGSynPlas, Neuron, postProcessing, cpg_rbfn

M_PI=3.1415

data = {}
with open("./data/RL_job_base.json") as file:
    data = json.load(file)

weights = [data["ParameterSet"]]
sensorWeights = [data["SensorParameterSet"]]
encoding = data["checked"]

CPG_RBFN = cpg_rbfn(weights, encoding, 20, "walk", sensorWeights)
startPhi = 0.015 * M_PI
CPG_RBFN.setPhii(startPhi)

CPGPeriodPostprocessor = postProcessing()

# Calculate init period
# We use random number to let the CPG start at different time = improved robustness
tau = 1
randomNumber = int(rand.uniform(1, tau) + 1)  #Random number between 1 and tau // was = 1
for i in range(0,tau+randomNumber):
    CPGPeriodPostprocessor.calculateAmplitude(CPG_RBFN.getCpgOutput(0), CPG_RBFN.getCpgOutput(1))
    CPGPeriod = int(CPGPeriodPostprocessor.getPeriod())
    CPG_RBFN.setCPGPeriod(CPGPeriod)

    sensorOutput = []
    CPG_RBFN.step(sensorOutput)

    RBF_output = CPG_RBFN.getNetworkOutput()

    print(RBF_output)