// pywrap.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "ann-framework/ann.h"
#include "ann-framework/backpropagation.h"
#include "ann-framework/circann.h"
#include "ann-framework/neuron.h"
#include "ann-framework/synapse.h"
#include "ann-framework/transferfunction.h"
#include "ann-library/so2cpg.h"
#include "ann-library/pcpg.h"
#include "ann-library/extendedso2cpg.h"
#include "ann-library/adaptiveso2cpgsynplas.h"
#include "controllers/postProcessing.h"
#include "controllers/rbfn.h"
#include "controllers/cpg_rbfn.h"
#include "delayline.h"

namespace py = pybind11;

PYBIND11_MODULE(ann_lib, m) {

    py::class_<Neuron>(m, "Neuron")
    .def(py::init<>())
    .def("addSynapseIn", &Neuron::addSynapseIn)
    .def("addSynapseOut", &Neuron::addSynapseOut)
    .def("getActivity", &Neuron::getActivity)
    .def("getBias", &Neuron::getBias)
    .def("getError", &Neuron::getError)
    .def("getInput", &Neuron::getInput)
    .def("getInputScaling", &Neuron::getInputScaling)
    .def("getOutput", &Neuron::getOutput)
    .def("getSynapseFrom", &Neuron::getSynapseFrom)
    .def("getSynapsesIn", &Neuron::getSynapsesIn)
    .def("getSynapsesOut", &Neuron::getSynapsesOut)
    .def("getSynapseTo", &Neuron::getSynapseTo)
    .def("removeSynapseIn", &Neuron::removeSynapseIn)
    .def("removeSynapseOut", &Neuron::removeSynapseOut)
    .def("setActivity", &Neuron::setActivity)
    .def("setBias", &Neuron::setBias)
    .def("setErrorInput", &Neuron::setErrorInput)
    .def("setInput", &Neuron::setInput)
    .def("setInputScaling", &Neuron::setInputScaling)
    .def("setOutput", &Neuron::setOutput)
    .def("setTransferFunction", &Neuron::setTransferFunction)
    .def("updateActivity", &Neuron::updateActivity)
    .def("updateError", &Neuron::updateError)
    .def("updateOutput", &Neuron::updateOutput)
    ;

    py::class_<Synapse>(m, "Synapse")
    .def(py::init<Neuron * const, Neuron * const, const bool &>())
    .def("getDeltaWeight", &Synapse::getDeltaWeight)
    .def("getPost", &Synapse::getPost)
    .def("getPre", &Synapse::getPre)
    .def("getWeight", &Synapse::getWeight)
    .def("setDeltaWeight", &Synapse::setDeltaWeight)
    .def("setWeight", &Synapse::setWeight)
    .def("updateWeight", &Synapse::updateWeight)
    ;

    py::class_<ANN>(m, "ANN")
    .def(py::init<>())
    .def(py::init<int>())
    .def("backpropagationStep", &ANN::backpropagationStep)
    .def("dumpBiases", &ANN::dumpBiases)
    .def("dumpWeights", &ANN::dumpWeights)
    .def("dw", py::overload_cast<const int &, const int &, const double &>(&ANN::dw))
    .def("dw", py::overload_cast<Neuron *, Neuron *, const double &>(&ANN::dw))
    .def("dw", py::overload_cast<const int &, const int &>(&ANN::dw))
    .def("feedForwardStep", &ANN::feedForwardStep)
    .def("getActivity", py::overload_cast<const int>(&ANN::getActivity, py::const_))
    .def("getActivity", py::overload_cast<Neuron const *>(&ANN::getActivity))
    .def("getAllNeurons", &ANN::getAllNeurons)
    .def("getAllSynapses", &ANN::getAllNeurons)
    .def("getBias", py::overload_cast<const int>(&ANN::getBias, py::const_))
    .def("getBias", py::overload_cast<Neuron const *>(&ANN::getBias))
    .def("getDefaultTransferFunction", &ANN::getDefaultTransferFunction)
    .def("getDeltaWeight", py::overload_cast<const int &, const int &>(&ANN::getDeltaWeight, py::const_))
    .def("getDeltaWeight", py::overload_cast<Neuron const *, Neuron const *>(&ANN::getDeltaWeight, py::const_))
    .def("getInput", py::overload_cast<const int>(&ANN::getInput, py::const_))
    .def("getInput", py::overload_cast<Neuron const *>(&ANN::getInput))
    .def("getInputScaling", py::overload_cast<const int>(&ANN::getInputScaling, py::const_))
    .def("getInputScaling", py::overload_cast<Neuron const *>(&ANN::getInputScaling))
    .def("getNeuron", &ANN::getNeuron)
    .def("getNeuronNumber", &ANN::getNeuronNumber)

    .def("getOutput", py::overload_cast<const int>(&ANN::getOutput, py::const_))
    .def("getOutput", py::overload_cast<Neuron const *>(&ANN::getOutput))

    .def("getSubnet", &ANN::getSubnet)

    .def("getSynapse", py::overload_cast<const unsigned int &, const unsigned int &>(&ANN::getSynapse, py::const_))
    .def("getSynapse", py::overload_cast<Neuron const * const, Neuron const * const>(&ANN::getSynapse))

    .def("getTopologicalSort", &ANN::getTopologicalSort)
    .def("getTotalNeuronNumber", &ANN::getTotalNeuronNumber)

    .def("getWeight", py::overload_cast<const int &, const int &>(&ANN::getWeight, py::const_))
    .def("getWeight", py::overload_cast<Neuron const *, Neuron const *>(&ANN::getWeight, py::const_))

    .def_static("identityFunction", &ANN::identityFunction)
    .def_static("linthresholdFunction", &ANN::linthresholdFunction)
    .def_static("logisticFunction", &ANN::logisticFunction)
    .def("N", &ANN::N)
    .def("n", &ANN::n)
    .def("postProcessing", &ANN::postProcessing)
    .def("removeNeuron", &ANN::removeNeuron)

    .def("setActivity", py::overload_cast<const int &, const double &>(&ANN::setActivity))
    .def("setActivity", py::overload_cast<Neuron *, const double &>(&ANN::setActivity))

    .def("setAllTransferFunctions", &ANN::setAllTransferFunctions)

    .def("setBias", py::overload_cast<Neuron *, const double &>(&ANN::setBias))
    .def("setBias", py::overload_cast<const int &, const double &>(&ANN::setBias))

    .def("setDefaultTransferFunction", &ANN::setDefaultTransferFunction)

    .def("setDeltaWeight", py::overload_cast<Neuron *, Neuron *, const double>(&ANN::setDeltaWeight))
    .def("setDeltaWeight", py::overload_cast<const int, const int, const double>(&ANN::setDeltaWeight))

    .def("setInput", py::overload_cast<const int &, const double &>(&ANN::setInput))
    .def("setInput", py::overload_cast<Neuron *, const double>(&ANN::setInput))

    .def("setInputScaling", py::overload_cast<const int &, const double &>(&ANN::setInputScaling))
    .def("setInputScaling", py::overload_cast<Neuron *, const double>(&ANN::setInputScaling))

    .def("setOutput", py::overload_cast<const int &, const double &>(&ANN::setOutput))
    .def("setOutput", py::overload_cast<Neuron *, const double>(&ANN::setOutput))

    .def("setTransferFunction", py::overload_cast<const int, TransferFunction const * const>(&ANN::setTransferFunction))
    .def("setTransferFunction", py::overload_cast<Neuron *, TransferFunction const * const>(&ANN::setTransferFunction))

    .def("setWeight", py::overload_cast<Neuron*, Neuron*, const double>(&ANN::setWeight))
    .def("setWeight", py::overload_cast<const int, const int, const double>(&ANN::setWeight))

    .def("signFunction", &ANN::signFunction)
    .def("step", &ANN::step)
    .def_static("tanhFunction", &ANN::tanhFunction)
    .def_static("thresholdFunction", &ANN::thresholdFunction)
    .def("updateActivities", &ANN::updateActivities)
    .def("updateOutputs", &ANN::updateOutputs)
    .def("updateTopologicalSort", &ANN::updateTopologicalSort)
    .def("updateWeights", &ANN::updateWeights)
    .def("setNeuronNumber", &ANN::setNeuronNumber)
    .def("addNeuron", &ANN::addNeuron)
    .def("addSubnet", &ANN::addSubnet)
    .def("addSynapse", &ANN::addSynapse)

    .def("b", py::overload_cast<const int, const double &>(&ANN::b))
    .def("b", py::overload_cast<Neuron *, const double &>(&ANN::b))
    .def("b", py::overload_cast<const int>(&ANN::b))

    .def("w", py::overload_cast<const int &, const int &, const double &>(&ANN::w))
    .def("w", py::overload_cast<Neuron*, Neuron*, const double &>(&ANN::w))
    ;

    py::class_<PCPG, ANN>(m, "PCPG")
    .def(py::init<>())
    .def("updateOutputs", &PCPG::updateOutputs)
    ;

    py::class_<SO2CPG, ANN>(m, "SO2CPG")
    .def(py::init<>())
    .def("enableFrequencyTable", &SO2CPG::enableFrequencyTable)
    .def("getAlpha", &SO2CPG::getAlpha)
    .def("getFrequency", &SO2CPG::getFrequency)
    .def("getPhi", py::overload_cast<>(&SO2CPG::getPhi, py::const_))
    .def("getPhi", py::overload_cast<const double &>(&SO2CPG::getPhi, py::const_))
    .def("setAlpha", &SO2CPG::setAlpha)
    .def("setFrequency", &SO2CPG::setFrequency)
    .def("setPhi", &SO2CPG::setPhi)
    .def("getAlpha", &SO2CPG::getAlpha)
    .def("updateFrequencyTable", &SO2CPG::updateFrequencyTable)
    .def("updateSO2Weights", &SO2CPG::updateSO2Weights)
    ;

    py::class_<ExtendedSO2CPG, SO2CPG>(m, "ExtendedSO2CPG")
    .def(py::init<Neuron *>(), py::arg("perturbingNeuron") = 0)
    .def("allowResets", &ExtendedSO2CPG::allowResets)
    .def("getBeta", &ExtendedSO2CPG::getBeta)
    .def("getEpsilon", &ExtendedSO2CPG::getEpsilon)
    .def("getGamma", &ExtendedSO2CPG::getGamma)
    .def("getMu", &ExtendedSO2CPG::getMu)
    .def("getPerturbation", &ExtendedSO2CPG::getPerturbation)
    .def("getPerturbingNeuron", &ExtendedSO2CPG::getPerturbingNeuron)
    .def("setBeta", &ExtendedSO2CPG::setBeta)
    .def("setGamma", &ExtendedSO2CPG::setGamma)
    .def("setEpsilon", &ExtendedSO2CPG::setEpsilon)
    .def("setMu", &ExtendedSO2CPG::setMu)
    .def("setPerturbation", &ExtendedSO2CPG::setPerturbation)
    .def("postProcessing", &ExtendedSO2CPG::postProcessing)
    .def("shouldReset", &ExtendedSO2CPG::shouldReset)
    .def("reset", &ExtendedSO2CPG::reset)
    ;

    py::class_<AdaptiveSO2CPGSynPlas, ExtendedSO2CPG>(m, "AdaptiveSO2CPGSynPlas")
    .def(py::init<>())
    .def("updateWeights", &AdaptiveSO2CPGSynPlas::updateWeights)
    .def("setBetaDynamics", &AdaptiveSO2CPGSynPlas::setBetaDynamics)
    .def("setGammaDynamics", &AdaptiveSO2CPGSynPlas::setGammaDynamics)
    .def("setEpsilonDynamics", &AdaptiveSO2CPGSynPlas::setEpsilonDynamics)
    ;

    py::class_<rbfn>(m, "rbfn")
    .def(py::init<int, vector<vector<float>>, string, string, vector<vector<float>>>())
    .def("getNumKernels", &rbfn::getNumKernels)
    .def("setCPGPeriod", &rbfn::setCPGPeriod)
    .def("setBeta", &rbfn::setBeta)
    .def("setWeights", &rbfn::setWeights)
    .def("setCenters", py::overload_cast<vector<float>, vector<float>>(&rbfn::setCenters))
    .def("setCenters", py::overload_cast<int>(&rbfn::setCenters))
    .def("getBeta", &rbfn::getBeta)
    .def("getWeights", &rbfn::getWeights)
    .def_readwrite("contributions", &rbfn::contributions)
    .def("step", &rbfn::step)
    .def("calculateCenters", &rbfn::calculateCenters)
    ;

    py::class_<postProcessing>(m, "postProcessing")
    .def(py::init<>())
    .def("getLPFSignal", &postProcessing::getLPFSignal)
    .def("getAmplitudeSignal", &postProcessing::getAmplitudeSignal)
    .def("calculateAmplitudeSignal", &postProcessing::calculateAmplitudeSignal)
    .def("calculateLPFSignal", &postProcessing::calculateLPFSignal)
    .def("calculateLPFAmplitude", &postProcessing::calculateLPFAmplitude)
    .def("calculateAmplitude", &postProcessing::calculateAmplitude)
    .def("getTimeBetweenZeroDerivative", &postProcessing::getTimeBetweenZeroDerivative)
    .def("getPeriod", &postProcessing::getPeriod)
    .def("getSignalPeriod", &postProcessing::getSignalPeriod)
    .def_readwrite("periodViz", &postProcessing::periodViz)
    .def_readwrite("periodTrust", &postProcessing::periodTrust)
    ;

    py::class_<cpg_rbfn>(m, "cpg_rbfn")
    .def(py::init<vector<vector<float>>, string, int, string, vector<vector<float>>>())
    .def("setCPGPeriod", &cpg_rbfn::setCPGPeriod)
    .def("step", &cpg_rbfn::step)
    .def("getCpgOutput", &cpg_rbfn::getCpgOutput)
    .def("getCpgActivity", &cpg_rbfn::getCpgActivity)

    .def_readwrite("MI", &cpg_rbfn::MI)
    .def_readwrite("cpg_bias", &cpg_rbfn::cpg_bias)

    .def("getCpgWeight", &cpg_rbfn::getCpgWeight)
    .def("getCpgBias", &cpg_rbfn::getCpgBias)
    .def("setPerturbation", &cpg_rbfn::setPerturbation)
    .def("getPhi", &cpg_rbfn::getPhi)
    .def("setPhii", &cpg_rbfn::setPhii)
    .def("calculateRBFCenters", &cpg_rbfn::calculateRBFCenters)
    .def("getNetworkOutput", &cpg_rbfn::getNetworkOutput)
    .def("getContribution", &cpg_rbfn::getContribution)
    ;

    py::class_<Delayline>(m, "Delayline")
    .def(py::init<int>())
    .def("Read", &Delayline::Read)
    .def("Write", &Delayline::Write)
    .def("Step", &Delayline::Step)
    .def("Reset", &Delayline::Reset)
    .def_static("mod", &Delayline::mod)
    .def_readwrite("buffer", &Delayline::buffer)
    .def_readwrite("step", &Delayline::step)
    ;
}