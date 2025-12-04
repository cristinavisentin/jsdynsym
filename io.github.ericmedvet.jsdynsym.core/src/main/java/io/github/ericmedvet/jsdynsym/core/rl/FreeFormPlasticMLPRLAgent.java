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

  private static final String AVERAGE = "AVERAGE";
  private static final String STD_DEV = "STD_DEV";
  private static final String CURRENT = "CURRENT";
  private static final String TREND = "TREND";
  private static final String ACTIVATION = "ACTIVATION";
  private static final String PRE = "-PRE_SYNAPTIC";
  private static final String POST = "-POST_SYNAPTIC";
  private static final String NEURON = "-NEURON-";
  private static final String LAYER = "-LAYER-";
  private static final String NETWORK = "-NETWORK-";
  private static final String REWARD = "REWARD";
  private static final String AGE = "AGE";
  private static final String PRE_SYNAPTIC_NEURON_INDEX = "PRE_SYNAPTIC_NEURON_INDEX";
  private static final String POST_SYNAPTIC_NEURON_INDEX = "POST_SYNAPTIC_NEURON_INDEX";
  private static final String LAYER_INDEX = "LAYER_INDEX";

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

  private static double[][] computeOutput(
      double[] input,
      int[] neurons,
      MultiLayerPerceptron.ActivationFunction activationFunction,
      double[][][] weights,
      double[][] activations
  ) {
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
    double[][][] newWeights = state.weights;

    if (age > 0) {
      Map<String, Double> inputParameters = new HashMap<>();

      // compute statistics network wise
      inputParameters.put(AGE, (double) age);
      inputParameters.put(REWARD, reward);
      Statistics statsNetwork = Statistics.compute(networkHistory(state.activationsHistory), age);
      Statistics.insert(inputParameters, statsNetwork, Statistics.STATISTICS_CASE.NETWORK);

      // update weights
      for (int i = 1; i < neurons.length; i++) {
        // compute statistics layer wise
        Statistics statsLayerPost = Statistics.compute(layerHistory(state.activationsHistory, i), age);
        Statistics.insert(inputParameters, statsLayerPost, Statistics.STATISTICS_CASE.LAYER_POST);
        Statistics statsLayerPre = Statistics.compute(layerHistory(state.activationsHistory, i - 1), age);
        Statistics.insert(inputParameters, statsLayerPre, Statistics.STATISTICS_CASE.LAYER_PRE);

        for (int j = 0; j < newWeights[i].length; j++) {
          // compute statistics post synapse
          Statistics statsNeuronPost = Statistics.compute(neuronHistory(state.activationsHistory, i, j), age);
          Statistics.insert(inputParameters, statsNeuronPost, Statistics.STATISTICS_CASE.NEURON_POST);

          for (int k = 1; k < newWeights[i - 1][j].length; k++) {
            // compute statistics pre synapse
            Statistics statsNeuronPre = Statistics.compute(neuronHistory(state.activationsHistory, i - 1, k - 1), age);
            Statistics.insert(inputParameters, statsNeuronPre, Statistics.STATISTICS_CASE.NEURON_PRE);

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
        neurons,
        activationFunction,
        newWeights,
        age > 0 ? state.activationsHistory[(int) ((age - 1) % historyLength)] : new double[neurons.length][]
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
    //    public Statistics(double[] history, long age) {
    //      this(
    //          Arrays.stream(history).average().orElse(0),
    //          Math.sqrt(
    //              Arrays.stream(history)
    //                  .map(v -> Math.pow(v - Arrays.stream(history).average().orElse(0), 2))
    //                  .average().orElse(0)
    //          ),
    //          history[(int) age % history.length],
    //          history[(int) age % history.length] - history[(int) (age - 1) % history.length]// ultimo meno penultimo
    //      );
    //    }

    public static Statistics compute(double[] history, long age) {
      if (age == 0) {
        return new Statistics(0, 0, 0, 0);
      }
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

    public static void insert(Map<String, Double> container, Statistics statistics, STATISTICS_CASE statisticsCase) {
      switch (statisticsCase) {
        case NETWORK -> {
          container.put(AVERAGE + NETWORK + ACTIVATION, statistics.average);
          container.put(STD_DEV + NETWORK + ACTIVATION, statistics.stdDev);
          container.put(CURRENT + NETWORK + ACTIVATION, statistics.current);
          container.put(TREND + NETWORK + ACTIVATION, statistics.trend);
        }
        case LAYER_PRE -> {
          container.put(AVERAGE + PRE + LAYER + ACTIVATION, statistics.average);
          container.put(STD_DEV + PRE + LAYER + ACTIVATION, statistics.stdDev);
          container.put(CURRENT + PRE + LAYER + ACTIVATION, statistics.current);
          container.put(TREND + PRE + LAYER + ACTIVATION, statistics.trend);
        }
        case LAYER_POST -> {
          container.put(AVERAGE + POST + LAYER + ACTIVATION, statistics.average);
          container.put(STD_DEV + POST + LAYER + ACTIVATION, statistics.stdDev);
          container.put(CURRENT + POST + LAYER + ACTIVATION, statistics.current);
          container.put(TREND + POST + LAYER + ACTIVATION, statistics.trend);
        }
        case NEURON_PRE -> {
          container.put(AVERAGE + PRE + NEURON + ACTIVATION, statistics.average);
          container.put(STD_DEV + PRE + NEURON + ACTIVATION, statistics.stdDev);
          container.put(CURRENT + PRE + NEURON + ACTIVATION, statistics.current);
          container.put(TREND + PRE + NEURON + ACTIVATION, statistics.trend);
        }
        case NEURON_POST -> {
          container.put(AVERAGE + POST + NEURON + ACTIVATION, statistics.average);
          container.put(STD_DEV + POST + NEURON + ACTIVATION, statistics.stdDev);
          container.put(CURRENT + POST + NEURON + ACTIVATION, statistics.current);
          container.put(TREND + POST + NEURON + ACTIVATION, statistics.trend);
        }
      }
    }

    private enum STATISTICS_CASE {
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
