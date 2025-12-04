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

  private static final String AVERAGE = "AVG";
  private static final String STDDEV = "STDDEV";
  private static final String CURRENT = "CURRENT";
  private static final String TREND = "TREND";
  private static final String ACTIVATION = "ACTIVATION";
  private static final String PRE = "PRE";
  private static final String POST = "POST";
  private static final String NEURON_WISE = "NEURON_WISE";
  private static final String LAYER_WISE = "LAYER_WISE";
  private static final String NETWORK_WISE = "NETWORK_WISE";

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
    for (int i = 0; i < historyLength; i++) {
      emptyActivations[i] = new double[neurons.length][];
      for (int j = 0; j < neurons.length; j++) {
        emptyActivations[i][j] = new double[neurons[i]];
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

    if (age == 0) {
      double[][][] weights = state.weights;
      double[][] activationValues = computeOutput(
          input,
          neurons,
          activationFunction,
          weights,
          new double[neurons.length][]
      );

      double[][][] activationsHistory = state.activationsHistory;
      activationsHistory[0] = activationValues;

      double[] rewardsHistory = state.rewardsHistory;
      rewardsHistory[0] = reward;

      state = new State(1, weights, activationsHistory, rewardsHistory);
      return activationValues[neurons.length - 1];
    }

    Map<String, Double> inputParameters = new HashMap<>();
    int historyIndexRead = (int) ((age - 1) % historyLength);
    int historyIndexWrite = (int) (age % historyLength);

    // compute statistics network wise
    inputParameters.put(AGE, (double) age);
    inputParameters.put(REWARD, reward);
    Statistics snw = Statistics.compute(networkHistory(state.activationsHistory), age);
    inputParameters.put(AVERAGE + ACTIVATION + NETWORK_WISE, snw.average);
    inputParameters.put(TREND + ACTIVATION + NETWORK_WISE, snw.trend);
    inputParameters.put(CURRENT + ACTIVATION + NETWORK_WISE, snw.current);
    inputParameters.put(STDDEV + ACTIVATION + NETWORK_WISE, snw.stdDev);

    // update weights
    double[][][] newWeights = state.weights;
    for (int i = 1; i < neurons.length; i++) {
      // compute statistics layer wise
      Statistics slwpre = Statistics.compute(layerHistory(state.activationsHistory, i), age);
      Statistics slwpost = Statistics.compute(layerHistory(state.activationsHistory, i - 1), age);
      inputParameters.put(AVERAGE + PRE + ACTIVATION + LAYER_WISE, slwpre.average);
      inputParameters.put(TREND + PRE + ACTIVATION + LAYER_WISE, slwpre.trend);
      inputParameters.put(CURRENT + PRE + ACTIVATION + LAYER_WISE, slwpre.current);
      inputParameters.put(STDDEV + PRE + ACTIVATION + LAYER_WISE, slwpre.stdDev);
      inputParameters.put(AVERAGE + POST + ACTIVATION + LAYER_WISE, slwpost.average);
      inputParameters.put(TREND + POST + ACTIVATION + LAYER_WISE, slwpost.trend);
      inputParameters.put(CURRENT + POST + ACTIVATION + LAYER_WISE, slwpost.current);
      inputParameters.put(STDDEV + POST + ACTIVATION + LAYER_WISE, slwpost.stdDev);

      for (int j = 0; j < newWeights[i - 1].length; j++) {
        // compute statistics post synapse
        Statistics sneuwpost = Statistics.compute(neuronHistory(state.activationsHistory, i, j), age);
        inputParameters.put(AVERAGE + PRE + ACTIVATION + NEURON_WISE, sneuwpost.average);
        inputParameters.put(TREND + PRE + ACTIVATION + NEURON_WISE, sneuwpost.trend);
        inputParameters.put(CURRENT + PRE + ACTIVATION + NEURON_WISE, sneuwpost.current);
        inputParameters.put(STDDEV + PRE + ACTIVATION + NEURON_WISE, sneuwpost.stdDev);

        for (int k = 1; k < newWeights[i - 1][j].length; k++) {
          // compute statistics pre synapse
          Statistics sneuwpre = Statistics.compute(neuronHistory(state.activationsHistory, i - 1, k - 1), age);
          inputParameters.put(AVERAGE + POST + ACTIVATION + NEURON_WISE, sneuwpre.average);
          inputParameters.put(TREND + POST + ACTIVATION + NEURON_WISE, sneuwpre.trend);
          inputParameters.put(CURRENT + POST + ACTIVATION + NEURON_WISE, sneuwpre.current);
          inputParameters.put(STDDEV + POST + ACTIVATION + NEURON_WISE, sneuwpre.stdDev);

          inputParameters.put(LAYER_INDEX, (double) i);
          inputParameters.put(PRE_SYNAPTIC_NEURON_INDEX, (double) k - 1);
          inputParameters.put(POST_SYNAPTIC_NEURON_INDEX, (double) j);

          newWeights[i - 1][j][k] += plasticityFunction.computeAsDouble(inputParameters);
        }
      }
    }

    // compute output
    double[][] newActivations = computeOutput(
        input,
        neurons,
        activationFunction,
        newWeights,
        state.activationsHistory[historyIndexRead]
    );

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
              .average().orElse(0)
      );
      double current = history[(int) age % history.length];
      double trend = current - history[(int) (age - 1) % history.length]; // ultimo meno penultimo
      return new Statistics(current, trend, avg, stdDev);
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
