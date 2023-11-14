# Utils

This module contains all project-wide uitility functions.

If any utility is used in multiple modules, it should be placed here. Otherwise, it should be placed in the module that uses it.

We have converted the ANN-Library and the CPG-RBFN class from the [reference code](https://github.com/MathiasThor/CPG-RBFN-framework) into python module. We are not inventing anything new in the base network architecture used by M.Thor in CPG-RBFN framework. We are just using the same architecture to extend it to Evolutionary RL for learning parameters of the actor networks.

The python bindings are placed in this directory and can be imported as ann_lib. The bindings are generated using pybind11. This reduces our work in the network part of the project.

Good practice to import the module is to use the following code snippet:

```python
import os
import sys

# This will add the project root path for a file placed in any subfolder of the project directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import utils.ann_lib as ann
```

Change the path insert if creating a file in a sub-folder within a sub-folder of the project.

## Classes binded

- Neuron
- Synapse
- PostProcessing
- ANN
- PCPG
- SO2CPG
- ExtendedSO2CPG
- AdaptiveSO2CPGSynPlas
- RBFN
- CPG-RBFN
