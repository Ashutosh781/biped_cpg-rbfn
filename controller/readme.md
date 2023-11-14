# Controller

This module contains the controller for the robot. The controller is based on the CPG-RBFN framework by M.Thor. The CPG-RBFN framework is extended to include the evolutionary RL algorithm for learning the parameters of the actor networks.

The controller class created interface between the network and the simulation environment. It uses the network from the python bindings in utils module and the simulation environment from the sim module.
