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
import java.util.stream.Collectors;

public class FreeFormPlasticMLPRLAgent implements NumericalTimeInvariantReinforcementLearningAgent<FreeFormPlasticMLPRLAgent.State>, Parametrized<FreeFormPlasticMLPRLAgent, NamedUnivariateRealFunction>, FrozenableNumericalRLAgent<FreeFormPlasticMLPRLAgent.State> {
  private static final String AVERAGE = "Average";
  private static final String STD_DEV = "Std_Dev";
  private static final String CURRENT = "Current";
  private static final String TREND = "Trend";
  private static final String ACTIVATION = "Activation";
  private static final String REWARD = "Reward";
  private static final String AGE = "Age";
  private static final String PRE_SYNAPTIC_NEURON_INDEX = "Pre-Synaptic Neuron Idx";
  private static final String POST_SYNAPTIC_NEURON_INDEX = "Post-Synaptic Neuron Idx";
  private static final String LAYER_INDEX = "Layer Idx";
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
    for (int h = 0; h < historyLength; h++) {
      emptyActivations[h] = new double[neurons.length][];
      for (int i = 0; i < neurons.length; i++) {
        emptyActivations[h][i] = new double[neurons[i]];
      }
    }
    return emptyActivations;
  }


  private static double[] networkHistory(double[][][] activationsHistory) {
    return Arrays.stream(activationsHistory)
        .mapToDouble(
            h -> Arrays.stream(h)
                .flatMapToDouble(Arrays::stream)
                .average()
                .orElse(0)
        )
        .toArray();
  }

  private static double[] layerHistory(double[][][] activationsHistory, int layerIdx) {
    return Arrays.stream(activationsHistory)
        .mapToDouble(
            a -> Arrays.stream(a[layerIdx])
                .average()
                .orElse(0)
        )
        .toArray();
  }

  private static double[] neuronHistory(double[][][] activationsHistory, int layerIdx, int neuronIdx) {
    return Arrays.stream(activationsHistory).mapToDouble(a -> a[layerIdx][neuronIdx]).toArray();
  }

  private static double[][] computeOutput(
      double[] input,
      double[][][] weights,
      MultiLayerPerceptron.ActivationFunction activationFunction,
      double[][] activations
  ) {
    activations[0] = Arrays.stream(input).map(activationFunction).toArray();
    for (int i = 1; i < activations.length; i++) {
      activations[i] = new double[weights[i - 1].length];
      for (int j = 0; j < activations[i].length; j++) {
        double sum = weights[i - 1][j][0];
        for (int k = 1; k < weights[i - 1][j].length; k++) {
          sum = sum + activations[i - 1][k - 1] * weights[i - 1][j][k];
        }
        activations[i][j] = activationFunction.applyAsDouble(sum);
      }
    }
    return activations;
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
  public double[] step(double[] input, double reward) {
    if (input.length != neurons[0]) {
      throw new IllegalArgumentException(
          String.format("Expected input length is %d: found %d", neurons[0], input.length)
      );
    }
    long age = state.age;
    double[][][] newWeights = state.weights;

    if (age > 0) {
      Map<String, Double> inputParameters = new HashMap<>();

      // compute statistics network wise
      inputParameters.put(AGE, (double) age);
      inputParameters.put(REWARD, reward);
      Statistics.from(networkHistory(state.activationsHistory), age)
          .insert(inputParameters, Statistics.StatisticsScope.NETWORK);

      // update weights
      for (int i = 1; i < neurons.length; i++) {
        // compute statistics layer wise
        Statistics.from(layerHistory(state.activationsHistory, i), age)
            .insert(inputParameters, Statistics.StatisticsScope.LAYER_POST);
        Statistics.from(layerHistory(state.activationsHistory, i - 1), age)
            .insert(inputParameters, Statistics.StatisticsScope.LAYER_PRE);

        for (int j = 0; j < newWeights[i].length; j++) {
          // compute statistics post synapse
          Statistics.from(neuronHistory(state.activationsHistory, i, j), age)
              .insert(inputParameters, Statistics.StatisticsScope.NEURON_POST);

          for (int k = 1; k < newWeights[i - 1][j].length; k++) {
            // compute statistics pre synapse
            Statistics.from(neuronHistory(state.activationsHistory, i - 1, k - 1), age)
                .insert(inputParameters, Statistics.StatisticsScope.NEURON_PRE);

            inputParameters.put(LAYER_INDEX, (double) i);
            inputParameters.put(PRE_SYNAPTIC_NEURON_INDEX, (double) (k - 1));
            inputParameters.put(POST_SYNAPTIC_NEURON_INDEX, (double) j);

            newWeights[i - 1][j][k] += plasticityFunction.computeAsDouble(inputParameters);
          }
        }
      }
    }

    // compute output
    double[][] newActivations = computeOutput(
        input,
        newWeights,
        activationFunction,
        new double[neurons.length][]
    );

    // update histories
    int historyIndex = (int) (age % historyLength);
    double[] updatedRewardsHistory = state.rewardsHistory;
    updatedRewardsHistory[historyIndex] = reward;
    double[][][] updatedActivationsHistory = state.activationsHistory;
    updatedActivationsHistory[historyIndex] = newActivations;

    // update state
    state = new State(age + 1, newWeights, updatedActivationsHistory, updatedRewardsHistory);

    // return network outputs
    return newActivations[neurons.length - 1];
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

  private record Statistics(
      double current,
      double trend,
      double average,
      double stdDev
  ) {

    public static Statistics from(double[] history, long age) {
      double avg = Arrays.stream(history).average().orElse(0);
      double stdDev = Math.sqrt(
          Arrays.stream(history)
              .map(v -> Math.pow(v - avg, 2))
              .average()
              .orElse(0)
      );
      double current = history[(int) age % history.length];
      double trend = current - history[(int) (age - 1) % history.length]; // ultimo - penultimo
      return new Statistics(current, trend, avg, stdDev);
    }

    public void insert(Map<String, Double> container, StatisticsScope statisticsScope) {
      String label = Arrays.stream(statisticsScope.toString().split("_"))
          .map(t -> t.charAt(0) + t.substring(1).toLowerCase())
          .collect(Collectors.joining(" ", " ", " "));
      container.put(AVERAGE + label + ACTIVATION, average);
      container.put(STD_DEV + label + ACTIVATION, stdDev);
      container.put(CURRENT + label + ACTIVATION, current);
      container.put(TREND + label + ACTIVATION, trend);
    }

    private enum StatisticsScope {
      NETWORK, LAYER_PRE, LAYER_POST, NEURON_PRE, NEURON_POST
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
