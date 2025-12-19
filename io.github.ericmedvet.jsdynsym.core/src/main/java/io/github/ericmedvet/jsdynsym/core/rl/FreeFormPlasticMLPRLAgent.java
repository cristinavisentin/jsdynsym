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
import io.github.ericmedvet.jsdynsym.core.numerical.NumericalStatelessSystem;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.HebbianMultiLayerPerceptron;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.MultiLayerPerceptron;
import io.github.ericmedvet.jsdynsym.core.numerical.named.NamedUnivariateRealFunction;
import java.util.*;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;

public class FreeFormPlasticMLPRLAgent implements NumericalTimeInvariantReinforcementLearningAgent<FreeFormPlasticMLPRLAgent.State>, Parametrized<FreeFormPlasticMLPRLAgent, NamedUnivariateRealFunction>, FrozenableNumericalRLAgent<FreeFormPlasticMLPRLAgent.State> {
  private static final String AVERAGE = "average";
  private static final String STD_DEV = "stdDev";
  private static final String CURRENT = "current";
  private static final String TREND = "trend";
  private static final String ACTIVATION = "activation";
  private static final String AGE = "age";
  private static final String PRE_SYNAPTIC_NEURON_INDEX = "preSynapticNeuronIdx";
  private static final String POST_SYNAPTIC_NEURON_INDEX = "postSynapticNeuronIdx";
  private static final String LAYER_INDEX = "layerIdx";
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
    double[][][] emptyActivations = new double[neurons.length][][];
    for (int i = 0; i < neurons.length; i++) {
      emptyActivations[i] = new double[neurons[i]][];
      for (int j = 0; j < neurons[i]; j++) {
        emptyActivations[i][j] = new double[historyLength];
      }
    }
    return emptyActivations;
  }

  private static StateAndOutput step(
      double[] input,
      double reward,
      State state,
      MultiLayerPerceptron.ActivationFunction activationFunction,
      NamedUnivariateRealFunction plasticityFunction,
      int[] neurons
  ) {
    long age = state.age;
    double[][][] newWeights = state.weights;
    if (age > 0) {
      Map<String, Double> inputParameters = new HashMap<>();
      // statistics network wise
      inputParameters.put(AGE, (double) age);
      Statistics.from(state.rewardsHistory, age - 1).insert(inputParameters, Statistics.StatisticsScope.REWARD);
      Statistics.from(state.networkHistory, age - 1).insert(inputParameters, Statistics.StatisticsScope.NETWORK);
      for (int i = 1; i < neurons.length; i++) {
        // statistics layer wise
        Statistics.from(state.layersHistory[i], age - 1)
            .insert(inputParameters, Statistics.StatisticsScope.LAYER_POST);
        Statistics.from(state.layersHistory[i - 1], age - 1)
            .insert(inputParameters, Statistics.StatisticsScope.LAYER_PRE);
        for (int j = 0; j < newWeights[i - 1].length; j++) {
          // statistics post synapse
          Statistics.from(state.activationsHistory[i][j], age - 1)
              .insert(inputParameters, Statistics.StatisticsScope.NEURON_POST);
          for (int k = 1; k < newWeights[i - 1][j].length; k++) {
            // statistics pre synapse
            Statistics.from(state.activationsHistory[i - 1][k - 1], age - 1)
                .insert(inputParameters, Statistics.StatisticsScope.NEURON_PRE);
            inputParameters.put(LAYER_INDEX, (double) i);
            inputParameters.put(PRE_SYNAPTIC_NEURON_INDEX, (double) k);
            inputParameters.put(POST_SYNAPTIC_NEURON_INDEX, (double) j);
            // update weights
            newWeights[i - 1][j][k] += plasticityFunction.computeAsDouble(inputParameters);
          }
        }
      }
    }
    // compute output
    double[][] activations = new double[neurons.length][];
    for (int i = 0; i < neurons.length; i++) {
      activations[i] = new double[neurons[i]];
    }
    double[][] newActivations = MultiLayerPerceptron.computeActivations(
        input,
        newWeights,
        activationFunction,
        activations
    );
    return new StateAndOutput(
        state.update(newActivations, reward),
        newActivations[neurons.length - 1]
    );
  }

  private static double[][][] deepCopy(double[][][] src, int historyLength, int[] neurons) {
    double[][][] copy = emptyActivations(historyLength, neurons);
    for (int i = 0; i < src.length; i++) {
      for (int j = 0; j < src[i].length; j++) {
        copy[i][j] = Arrays.copyOf(src[i][j], src[i][j].length);
      }
    }
    return copy;
  }

  public static List<String> getVariableNames() {
    String[] statisticTypes = {AVERAGE, STD_DEV, CURRENT, TREND};
    List<String> variableNames = new ArrayList<>();
    for (Statistics.StatisticsScope ss : Statistics.StatisticsScope.values()) {
      for (String st : statisticTypes) {
        if (ss.equals(Statistics.StatisticsScope.REWARD)) {
          variableNames.add(String.format("%s_%s", st, Statistics.StatisticsScope.REWARD));
        } else {
          variableNames.add(String.format("%s_%s_%s", st, ss, ACTIVATION));
        }
      }
    }
    variableNames.add(LAYER_INDEX);
    variableNames.add(POST_SYNAPTIC_NEURON_INDEX);
    variableNames.add(PRE_SYNAPTIC_NEURON_INDEX);
    variableNames.add(AGE);
    return variableNames;
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
  public FrozenableNumericalDynamicalSystem<?> dynamicalSystem() {
    final State initialState = new State(
        state.age,
        HebbianMultiLayerPerceptron.deepCopy(state.weights, neurons),
        deepCopy(state.activationsHistory, historyLength, neurons),
        Arrays.copyOf(state.rewardsHistory, historyLength),
        state.layersHistory, // TODO make a deep copy
        Arrays.copyOf(state.networkHistory, historyLength)
    );
    return new FrozenableNumericalDynamicalSystem<State>() {
      private State innerState = initialState;

      @Override
      public NumericalStatelessSystem stateless() {
        return new MultiLayerPerceptron(
            activationFunction,
            HebbianMultiLayerPerceptron.deepCopy(innerState.weights, neurons),
            neurons
        );
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
      public State getState() {
        return innerState;
      }

      @Override
      public void reset() {
        innerState = initialState;
      }

      @Override
      public double[] step(double t, double[] input) {
        StateAndOutput step = FreeFormPlasticMLPRLAgent.step(
            input,
            0,
            innerState,
            activationFunction,
            plasticityFunction,
            neurons
        );
        innerState = step.state;
        return step.output;
      }
    };
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
    StateAndOutput step = step(input, reward, state, activationFunction, plasticityFunction, neurons);
    state = step.state;
    return step.output;
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
        new double[historyLength],
        new double[emptyActivations(historyLength, neurons).length][historyLength],
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

  @Override
  public String toString() {
    return "Free-Form Plastic MLP RL Agent-%s-%s"
        .formatted(
            activationFunction.toString().toLowerCase(),
            Arrays.stream(neurons).mapToObj(Integer::toString).collect(Collectors.joining(">"))
        );
  }

  private record StateAndOutput(State state, double[] output) {
  }

  private record Statistics(
      double current,
      double trend,
      double average,
      double stdDev
  ) {

    private static Statistics from(double[] history, long age) {
      int currentIdx = (int) (age) % history.length;
      int oldestIdx = (age < history.length) ? 0 : (int) (age + 1) % history.length;
      double avg = 0;
      double numerator = 0;
      for (double v : history) {
        avg += v;
        numerator += Math.pow(v - avg, 2);
      }
      avg /= history.length;
      double stdDev = Math.sqrt(numerator / history.length);
      double current = history[currentIdx];
      double trend = current - history[oldestIdx]; // newest - oldest
      return new Statistics(current, trend, avg, stdDev);
    }

    private void insert(Map<String, Double> container, StatisticsScope statisticsScope) {
      if (statisticsScope.equals(StatisticsScope.REWARD)) {
        container.put(AVERAGE + "_" + statisticsScope, average);
        container.put(STD_DEV + "_" + statisticsScope, stdDev);
        container.put(CURRENT + "_" + statisticsScope, current);
        container.put(TREND + "_" + statisticsScope, trend);
      } else {
        container.put(AVERAGE + "_" + statisticsScope + "_" + ACTIVATION, average);
        container.put(STD_DEV + "_" + statisticsScope + "_" + ACTIVATION, stdDev);
        container.put(CURRENT + "_" + statisticsScope + "_" + ACTIVATION, current);
        container.put(TREND + "_" + statisticsScope + "_" + ACTIVATION, trend);
      }
    }

    private enum StatisticsScope {
      NEURON_POST("neuronPost"), NEURON_PRE("neuronPre"), LAYER_POST("layerPost"), LAYER_PRE("layerPre"), NETWORK(
          "network"
      ), REWARD("reward");

      private final String name;

      StatisticsScope(String name) {
        this.name = name;
      }

      @Override
      public String toString() {
        return name;
      }
    }
  }

  public record State(
      long age,
      double[][][] weights,
      double[][][] activationsHistory,
      double[] rewardsHistory,
      double[][] layersHistory, // history of average - layer wise - activation values
      double[] networkHistory // history of average - network wise - activation values
  ) {
    public State update(double[][] newActivations, double reward) {
      int nOfNeurons = 0;
      int historyIndex = (int) (age % rewardsHistory.length);
      networkHistory[historyIndex] = 0;
      for (int i = 0; i < newActivations.length; i++) {
        layersHistory[i][historyIndex] = 0;
        for (int j = 0; j < newActivations[i].length; j++) {
          activationsHistory[i][j][historyIndex] = newActivations[i][j];
          layersHistory[i][historyIndex] += newActivations[i][j];
          networkHistory[historyIndex] += newActivations[i][j];
        }
        layersHistory[i][historyIndex] /= activationsHistory[i].length;
        nOfNeurons += activationsHistory[i].length;
      }
      networkHistory[historyIndex] /= nOfNeurons;
      rewardsHistory[historyIndex] = reward;
      return new State(age + 1, weights, activationsHistory, rewardsHistory, layersHistory, networkHistory);
    }
  }
}
