/*-
 * ========================LICENSE_START=================================
 * jsdynsym-core
 * %%
 * Copyright (C) 2023 - 2025 Eric Medvet
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =========================LICENSE_END==================================
 */
package io.github.ericmedvet.jsdynsym.core.rl;

import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.Parametrized;
import io.github.ericmedvet.jsdynsym.core.numerical.FrozenableNumericalDynamicalSystem;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.HebbianMultiLayerPerceptron;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.MultiLayerPerceptron;
import io.github.ericmedvet.jsdynsym.core.numerical.named.NamedUnivariateRealFunction;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.random.RandomGenerator;

public class FreeFormPlasticMLPRLAgent implements NumericalTimeInvariantReinforcementLearningAgent<FreeFormPlasticMLPRLAgent.State>, Parametrized<FreeFormPlasticMLPRLAgent, NamedUnivariateRealFunction>, FrozenableNumericalRLAgent<FreeFormPlasticMLPRLAgent.State> {
  private final MultiLayerPerceptron.ActivationFunction activationFunction;
  private final int[] neurons;
  private final int historyLength;
  private final HebbianMultiLayerPerceptron.WeightInitializationType weightInitializationType;
  private final DoubleRange initialWeightRange;
  private final RandomGenerator randomGenerator;
  private NamedUnivariateRealFunction plasticityFunction;
  private State state;

  public FreeFormPlasticMLPRLAgent(
      MultiLayerPerceptron.ActivationFunction activationFunction,
      NamedUnivariateRealFunction plasticityFunction,
      int[] neurons,
      int historyLength,
      HebbianMultiLayerPerceptron.WeightInitializationType weightInitializationType,
      DoubleRange initialWeightRange,
      RandomGenerator randomGenerator
  ) {
    if (weightInitializationType.equals(HebbianMultiLayerPerceptron.WeightInitializationType.PARAMS)) {
      throw new IllegalArgumentException("Unsupported weight initialization type.");
    }
    this.activationFunction = activationFunction;
    this.plasticityFunction = plasticityFunction;
    this.neurons = neurons;
    this.historyLength = historyLength;
    this.weightInitializationType = weightInitializationType;
    this.initialWeightRange = initialWeightRange;
    this.randomGenerator = randomGenerator;
    reset();
  }

  public FreeFormPlasticMLPRLAgent(
      MultiLayerPerceptron.ActivationFunction activationFunction,
      NamedUnivariateRealFunction plasticityFunction,
      int nOfInput,
      int[] innerNeurons,
      int nOfOutput,
      int historyLength,
      HebbianMultiLayerPerceptron.WeightInitializationType weightInitializationType,
      DoubleRange initialWeightRange,
      RandomGenerator randomGenerator
  ) {
    this(
        activationFunction,
        plasticityFunction,
        MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput),
        historyLength,
        weightInitializationType,
        initialWeightRange,
        randomGenerator
    );
  }

  private static double[][][] emptyActivations(int historyLength, int[] neurons) {
    double[][][] emptyActivations = new double[historyLength][][];
    for (int i = 0; i < historyLength; i++) {
      emptyActivations[i] = new double[neurons.length][];
      for (int j = 0; j < neurons.length; j++) {
        emptyActivations[i][j] = new double[neurons[i]];
      }
    }
    return emptyActivations;
  }

  @Override
  public NamedUnivariateRealFunction getParams() {
    return plasticityFunction;
  }

  @Override
  public void setParams(NamedUnivariateRealFunction namedUnivariateRealFunction) {
    plasticityFunction = namedUnivariateRealFunction;
  }

  @Override
  public FrozenableNumericalDynamicalSystem<?> dynamicalSystem() { // TODO
    return null;
  }

  @Override
  public int nOfInputs() {
    return neurons[0];
  }

  @Override
  public int nOfOutputs() {
    return neurons[neurons.length - 1];
  }

  @Override
  public double[] step(double[] input, double reward) { // TODO
    if (input.length != neurons[0]) {
      throw new IllegalArgumentException(
          String.format("Expected input length is %d: found %d", neurons[0], input.length)
      );
    }
    long age = state.age;

    // age = 0
    // weights are zeros or random
    // histories are empty
    if (age == 0) {
      double[][][] weights = state.weights;
      double[][] activationValues = computeOutput(input, neurons, activationFunction, weights, new double[neurons.length][]);
//      activationValues[0] = Arrays.stream(input).map(activationFunction).toArray();
//      for (int i = 1; i < neurons.length; i++) {
//        activationValues[i] = new double[neurons[i]];
//        for (int j = 0; j < neurons[i]; j++) {
//          double sum = weights[i - 1][j][0];
//          for (int k = 1; k < neurons[i - 1] + 1; k++) {
//            sum = sum + activationValues[i - 1][k - 1] * weights[i - 1][j][k];
//          }
//          activationValues[i][j] = activationFunction.applyAsDouble(sum);
//        }
//      }
      double[][][] activationsHistory = state.activationsHistory;
      double[] rewardsHistory = state.rewardsHistory;
      activationsHistory[0] = activationValues;
      rewardsHistory[0] = reward;
      state = new State(1, weights, activationsHistory, rewardsHistory);
      return activationValues[neurons.length - 1];
    }

    // at age = n
    // I need statistics taken at age n-1
    int historyIndexRead = (int) ((age - 1) % historyLength);
    int historyIndexWrite = (int) (age % historyLength);

    // create the map... forse deve essere un field della classe?
    Map<String, Double> inputParameters = new HashMap<>();

    // update weights
    double[][][] newWeights = state.weights;

    // compute statistics network wise
    inputParameters.put("Agent-Age", (double) age);
    inputParameters.put("Reward", reward);
    inputParameters.put("Avg-Activation-Network", computeMeanNetworkWise(state.activationsHistory));
    // TODO add other statistics

    for (int i = 1; i < neurons.length; i++) {
      // compute statistics layer wise
      inputParameters.put("Avg-Activation-Layer", computeMeanLayerWise(state.activationsHistory, i));
      // TODO add other statistics

      for (int j = 0; j < newWeights[i - 1].length; j++) {
        inputParameters.put("Avg-PostActivation", computeMeanNeuronWise(state.activationsHistory, i, j));
        // TODO add other statistics
        for (int k = 1; k < newWeights[i - 1][j].length; k++) {
          inputParameters.put("Avg-PreActivation", computeMeanNeuronWise(state.activationsHistory, i - 1, k - 1));
          // TODO add other statistics

          inputParameters.put("Layer", (double) i);
          inputParameters.put("PreSynaptic-Neuron", (double) k - 1);
          inputParameters.put("PostSynaptic-Neuron", (double) j);

          newWeights[i - 1][j][k] += plasticityFunction.computeAsDouble(inputParameters);
        }
      }
    }

    // compute output
    //double[][] newActivations = state.activationsHistory[historyIndexRead];
    double[][] newActivations = computeOutput(input, neurons, activationFunction, newWeights, state.activationsHistory[historyIndexRead]);
//    newActivations[0] = Arrays.stream(input).map(activationFunction).toArray();
//    for (int i = 1; i < neurons.length; i++) {
//      newActivations[i] = new double[neurons[i]];
//      for (int j = 0; j < neurons[i]; j++) {
//        double sum = state.weights[i - 1][j][0];
//        for (int k = 1; k < neurons[i - 1] + 1; k++) {
//          sum = sum + newActivations[i - 1][k - 1] * state.weights[i - 1][j][k];
//        }
//        newActivations[i][j] = activationFunction.applyAsDouble(sum);
//      }
//    }

    // update rewards history
    double[] updatedRewardsHistory = state.rewardsHistory;
    updatedRewardsHistory[historyIndexWrite] = reward;

    // update activation history
    double[][][] updatedActivationsHistory = state.activationsHistory;
    updatedActivationsHistory[historyIndexWrite] = newActivations;

    // update state
    state = new State(age + 1, newWeights, updatedActivationsHistory, updatedRewardsHistory);

    // return network outputs
    return newActivations[neurons.length - 1];
  }

  private static double[][] computeOutput(double[] input, int[] neurons, MultiLayerPerceptron.ActivationFunction activationFunction, double[][][] weights, double[][] activations) {
    activations[0] = Arrays.stream(input).map(activationFunction).toArray();
    for (int i = 1; i < neurons.length; i++) {
      activations[i] = new double[neurons[i]];
      for (int j = 0; j < neurons[i]; j++) {
        double sum = weights[i - 1][j][0];
        for (int k = 1; k < neurons[i - 1] + 1; k++) {
          sum = sum + activations[i - 1][k - 1] * weights[i - 1][j][k];
        }
        activations[i][j] = activationFunction.applyAsDouble(sum);
      }
    }
    return activations;
  }

  private record ActivationStatistics(
      double current,
      double trend,
      double average,
      double stdDev
  ) {
  }

  private ActivationStatistics getNeuronStatistics(int layerIdx, int neuronIdx) {
    return new ActivationStatistics(
        getCurrentNeuronWise(state.activationsHistory(), layerIdx, neuronIdx),
        computeTrendNeuronWise(state.activationsHistory(), layerIdx, neuronIdx),
        computeMeanNeuronWise(state.activationsHistory(), layerIdx, neuronIdx),
        computeStdNeuronWise(state.activationsHistory(), layerIdx, neuronIdx)
    );
  }

  private double computeMeanNeuronWise(double[][][] activationsHistory, int layer, int neuron) {
    double sum = 0;
    for (double[][] activation : activationsHistory) {
      sum += activation[layer][neuron];
    }
    return sum / activationsHistory.length;
  }

  private double computeMeanLayerWise(double[][][] activationHistory, int layer) {
    double sum = 0;
    int nOfNeurons = neurons[layer];
    for (int n = 0; n < nOfNeurons; n++) {
      sum += computeMeanNeuronWise(activationHistory, layer, n);
    }
    return sum / nOfNeurons;
  }

  private double computeMeanNetworkWise(double[][][] activationHistory) {
    double sum = 0;
    int nOfLayers = neurons.length;
    for (int i = 1; i < nOfLayers; i++) {
      sum += computeMeanLayerWise(activationHistory, i);
    }
    return sum / (nOfLayers - 1);
  }

  private double computeStdNeuronWise(double[][][] activationsHistory, int layer, int neuron) {
    double mean = computeMeanNeuronWise(activationsHistory, layer, neuron);
    double squaredSum = 0;
    for (double[][] activation : activationsHistory) {
      double diff = activation[layer][neuron] - mean;
      squaredSum += diff * diff;
    }
    return Math.sqrt(squaredSum / activationsHistory.length);
  }

  private double computeTrendNeuronWise(double[][][] activationsHistory, int layer, int neuron) {
    double max = activationsHistory[0][layer][neuron];
    double min = activationsHistory[0][layer][neuron];

    for (int h = 1; h < activationsHistory.length; h++) {
      double value = activationsHistory[h][layer][neuron];
      if (value > max) {
        max = value;
      }
      if (value < min) {
        min = value;
      }
    }
    return max - min;
  }

  private double getCurrentNeuronWise(double[][][] activationsHistory, int layer, int neuron) {
    int historyIndex = Math.toIntExact(state.age % historyLength);
    return activationsHistory[historyIndex][layer][neuron];
  }

  @Override
  public State getState() {
    return state;
  }

  @Override
  public void reset() {
    state = new State(
        0,
        HebbianMultiLayerPerceptron.emptyArray(neurons),
        emptyActivations(historyLength, neurons),
        new double[historyLength]
    );
    if (weightInitializationType.equals(HebbianMultiLayerPerceptron.WeightInitializationType.RANDOM)) {
      for (int i = 1; i < neurons.length; i++) {
        for (int j = 0; j < neurons[i]; j++) {
          for (int k = 0; k < neurons[i - 1] + 1; k++) {
            state.weights[i - 1][j][k] = initialWeightRange.denormalize(randomGenerator.nextDouble());
          }
        }
      }
    }
  }

  public record State(
      long age,
      double[][][] weights,
      double[][][] activationsHistory,
      double[] rewardsHistory
  ) {
  }
}
