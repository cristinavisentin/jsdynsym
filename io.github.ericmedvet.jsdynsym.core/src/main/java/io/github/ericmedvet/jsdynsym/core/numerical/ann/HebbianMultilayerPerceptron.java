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
package io.github.ericmedvet.jsdynsym.core.numerical.ann;

import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.NumericalParametrized;
import io.github.ericmedvet.jsdynsym.core.FrozenableDynamicalSystem;
import io.github.ericmedvet.jsdynsym.core.numerical.FrozenableNumericalDynamicalSystem;
import io.github.ericmedvet.jsdynsym.core.numerical.NumericalStatelessSystem;
import io.github.ericmedvet.jsdynsym.core.numerical.NumericalTimeInvariantDynamicalSystem;

import java.util.Arrays;
import java.util.Random;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class HebbianMultilayerPerceptron implements NumericalTimeInvariantDynamicalSystem<HebbianMultilayerPerceptron.State>, NumericalParametrized<HebbianMultilayerPerceptron>, FrozenableNumericalDynamicalSystem<HebbianMultilayerPerceptron.State> {
  private static RandomGenerator randomGenerator;
  private final MultiLayerPerceptron.ActivationFunction activationFunction;
  private final double[][][] as;
  private final double[][][] bs;
  private final double[][][] cs;
  private final double[][][] ds;
  private final int[] neurons;
  private final double learningRate;
  private final DoubleRange initialWeightRange;
  private final ParametrizationType parametrizationType;
  private final WeightInitializationType weightInitializationType;
  private double[][][] initialWeights;
  private State state;

  public HebbianMultilayerPerceptron(
      MultiLayerPerceptron.ActivationFunction activationFunction,
      double[][][] as,
      double[][][] bs,
      double[][][] cs,
      double[][][] ds,
      int[] neurons,
      double learningRate, DoubleRange initialWeightRange,
      ParametrizationType parametrizationType,
      WeightInitializationType weightInitializationType
  ) {
    this.activationFunction = activationFunction;
    this.as = as;
    this.bs = bs;
    this.cs = cs;
    this.ds = ds;
    this.neurons = neurons;
    this.learningRate = learningRate;
    this.initialWeightRange = initialWeightRange;
    this.parametrizationType = parametrizationType;
    this.weightInitializationType = weightInitializationType;
    reset();
  }

  public HebbianMultilayerPerceptron(
      MultiLayerPerceptron.ActivationFunction activationFunction,
      int nOfInput,
      int[] innerNeurons,
      int nOfOutput,
      double[] params,
      double learningRate,
      DoubleRange initialWeightRange,
      ParametrizationType parametrizationType,
      WeightInitializationType weightInitializationType
  ) {
    this(
        activationFunction,
        initializeABCD(
            params,
            MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput),
            parametrizationType,
            ParamType.A
        ),
        initializeABCD(
            params,
            MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput),
            parametrizationType,
            ParamType.B
        ),
        initializeABCD(
            params,
            MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput),
            parametrizationType,
            ParamType.C
        ),
        initializeABCD(
            params,
            MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput),
            parametrizationType,
            ParamType.D
        ),
        MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput),
        learningRate,
        initialWeightRange,
        parametrizationType,
        weightInitializationType
    );
  }

  public HebbianMultilayerPerceptron(
      MultiLayerPerceptron.ActivationFunction activationFunction,
      int nOfInput,
      int[] innerNeurons,
      int nOfOutput,
      double learningRate,
      DoubleRange initialWeightRange,
      ParametrizationType parametrizationType,
      WeightInitializationType weightInitializationType
  ) {
    this(
        activationFunction,
        nOfInput,
        innerNeurons,
        nOfOutput,
        new double[countParams(weightInitializationType, parametrizationType, MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput))],
        learningRate,
        initialWeightRange,
        parametrizationType,
        weightInitializationType
    );
  }

  private static double[][][] initializeABCD(double[] params, int[] neurons, ParametrizationType parametrizationType, ParamType paramType) {
    int n = countParams(parametrizationType, neurons);
    return switch (paramType) {
      case A -> unflat(parametrizationType, Arrays.copyOfRange(params, 0, n), neurons);
      case B -> unflat(parametrizationType, Arrays.copyOfRange(params, n, 2 * n), neurons);
      case C -> unflat(parametrizationType, Arrays.copyOfRange(params, 2 * n, 3 * n), neurons);
      case D -> unflat(parametrizationType, Arrays.copyOfRange(params, 3 * n, 4 * n), neurons);
    };
  }

  private static int countParams(
      WeightInitializationType weightInitializationType,
      ParametrizationType parametrizationType,
      int[] neurons
  ) {
    return switch (weightInitializationType) {
      case ZEROS, RANDOM -> 4 * countParams(parametrizationType, neurons);
      case PARAMS -> 4 * countParams(parametrizationType, neurons) + MultiLayerPerceptron.countWeights(neurons);
    };
  }

  private static int countParams(ParametrizationType parametrizationType, int[] neurons) {
    return switch (parametrizationType) {
      case NETWORK -> 1;
      case LAYER -> (neurons.length - 1);
      case NEURON -> Arrays.stream(neurons).skip(1).sum();
      case SYNAPSE -> MultiLayerPerceptron.countWeights(neurons);
    };
  }

  private static double[][][] unflat(ParametrizationType parametrizationType, double[] params, int[] neurons) {
    if (params.length != countParams(parametrizationType, neurons)) {
      throw new IllegalArgumentException(
          String.format(
              "Wrong number of params: %d expected, %d found",
              countParams(parametrizationType, neurons),
              params.length
          )
      );
    }
    return switch (parametrizationType) {
      case NETWORK ->
          MultiLayerPerceptron.unflat(nCopies(MultiLayerPerceptron.countWeights(neurons), params[0]), neurons);
      case LAYER -> IntStream.range(1, neurons.length)
          .mapToObj(
              li -> IntStream.range(0, neurons[li])
                  .mapToObj(ni -> nCopies(neurons[li - 1] + 1, params[li - 1]))
                  .toArray(double[][]::new)
          )
          .toArray(double[][][]::new);
      case NEURON -> {
        yield neuronUnflat(params, neurons);
      }
      case SYNAPSE -> MultiLayerPerceptron.unflat(params, neurons);
    };
  }

  private static double[][][] neuronUnflat(double[] params, int[] neurons) {
    double[][][] unflat = unflat(ParametrizationType.NETWORK, new double[1], neurons);
    int c = 0;
    for (double[][] layer : unflat) {
      for (double[] neuron : layer) {
        Arrays.fill(neuron, params[c++]);
      }
    }
    return unflat;
  }

  private static double[] nCopies(int n, double value) {
    double[] values = new double[n];
    Arrays.fill(values, value);
    return values;
  }

  private static double[] flat(ParametrizationType parametrizationType, double[][][] params, int[] neurons) {
    return switch (parametrizationType) {
      case NETWORK -> new double[]{params[0][0][0]};
      case LAYER -> Arrays.stream(params)
          .mapToDouble(l -> l[0][0])
          .toArray();
      case NEURON -> Arrays.stream(params)
          .flatMap(l -> Arrays.stream(l).mapToDouble(n -> n[0]).boxed())
          .mapToDouble(v -> v)
          .toArray();
      case SYNAPSE -> MultiLayerPerceptron.flat(params, neurons);
    };
  }

  private static void set(double[][][] src, double[][][] dst) {
    for (int l = 0; l < src.length; l++) {
      for (int s = 0; s < src[l].length; s++) {
        System.arraycopy(src[l][s], 0, dst[l][s], 0, src[l][s].length);
      }
    }
  }

  private static double[] concat(double[]... arrays) {
    int totalLength = 0;
    for (double[] array : arrays) {
      totalLength += array.length;
    }
    double[] concatenated = new double[totalLength];
    int offset = 0;
    for (double[] array : arrays) {
      System.arraycopy(array, 0, concatenated, offset, array.length);
      offset += array.length;
    }
    return concatenated;
  }

  private static double[][][] initializeWeights(int[] neurons, WeightInitializationType weightInitializationType, DoubleRange range) {
    double[][][] initialWeights = new double[neurons.length - 1][][];
    for (int i = 1; i < neurons.length; i++) {
      initialWeights[i - 1] = new double[neurons[i]][neurons[i - 1] + 1];
    }
    switch (weightInitializationType) {
      case ZEROS, PARAMS -> {
      }
      case RANDOM -> {
        for (int i = 1; i < neurons.length; i++) {
          for (int j = 0; j < neurons[i]; j++) {
            for (int k = 0; k < neurons[i - 1] + 1; k++) {
              initialWeights[i - 1][j][k] = range.denormalize(randomGenerator.nextDouble());
            }
          }
        }
      }
    }
    ;
    return initialWeights;
  }

  @Override
  public double[] step(double[] input) {
    if (input.length != neurons[0]) {
      throw new IllegalArgumentException(
          String.format("Expected input length is %d: found %d", neurons[0], input.length)
      );
    }
    // update weights
    double[][][] newWeights = state.weights;
    for (int i = 1; i < neurons.length; i++) {
      for (int j = 0; j < newWeights[i - 1].length; j++) {
        double postActivation = state.activations[i][j];
        for (int k = 1; k < newWeights[i - 1][j].length; k++) {
          double preActivation = state.activations[i - 1][k - 1];
          newWeights[i - 1][j][k] += learningRate * (as[i - 1][j][k] * preActivation + bs[i - 1][j][k] * postActivation + cs[i - 1][j][k] * preActivation * postActivation + ds[i - 1][j][k]);
        }
      }
    }

    // compute output
    double[][] newActivations = state.activations;
    newActivations[0] = Arrays.stream(input).map(activationFunction).toArray();
    for (int i = 1; i < neurons.length; i++) {
      newActivations[i] = new double[neurons[i]];
      for (int j = 0; j < neurons[i]; j++) {
        double sum = state.weights[i - 1][j][0]; // set the bias
        for (int k = 1; k < neurons[i - 1] + 1; k++) {
          sum = sum + newActivations[i - 1][k - 1] * state.weights[i - 1][j][k];
        }
        newActivations[i][j] = activationFunction.applyAsDouble(sum);
      }
    }

    // update state
    state = new State(newWeights, newActivations);
    return newActivations[neurons.length - 1];
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
    return state;
  }

  @Override
  public void reset() {
    initialWeights = initializeWeights(neurons, weightInitializationType, initialWeightRange);
    state = new State(
        initialWeights,
        Arrays.stream(neurons).mapToObj(double[]::new).toArray(double[][]::new)
    );
  }

  @Override
  public double[] getParams() {
    double[] flatAs = flat(parametrizationType, as, neurons);
    double[] flatBs = flat(parametrizationType, bs, neurons);
    double[] flatCs = flat(parametrizationType, cs, neurons);
    double[] flatDs = flat(parametrizationType, ds, neurons);

    if (weightInitializationType.equals(WeightInitializationType.PARAMS)) {
      double[] flatWeights = MultiLayerPerceptron.flat(state.weights, neurons);
      return concat(flatAs, flatBs, flatCs, flatDs, flatWeights);
    } else {
      return concat(flatAs, flatBs, flatCs, flatDs);
    }
  }

  @Override
  public void setParams(double[] params) {
    int n = countParams(parametrizationType, neurons);
    set(unflat(parametrizationType, Arrays.copyOfRange(params, 0, n), neurons), as);
    set(unflat(parametrizationType, Arrays.copyOfRange(params, n, 2 * n), neurons), bs);
    set(unflat(parametrizationType, Arrays.copyOfRange(params, 2 * n, 3 * n), neurons), cs);
    set(unflat(parametrizationType, Arrays.copyOfRange(params, 3 * n, 4 * n), neurons), ds);
    if (weightInitializationType.equals(WeightInitializationType.PARAMS)) {
      set(
          unflat(ParametrizationType.SYNAPSE, Arrays.copyOfRange(params, 4 * n, params.length), neurons),
          state.weights
      );
    }
  }

  @Override
  public String toString() {
    return "HebbianMLP-%s-%s"
        .formatted(
            activationFunction.toString().toLowerCase(),
            Arrays.stream(neurons).mapToObj(Integer::toString).collect(Collectors.joining(">"))
        );
  }

  @Override
  public NumericalStatelessSystem stateless() {
    double[][][] frozenWeights = this.getState().weights;
    return new MultiLayerPerceptron(activationFunction, frozenWeights, neurons);
  }

  private enum ParamType {
    A, B, C, D
  }

  public enum ParametrizationType {
    NETWORK, LAYER, NEURON, SYNAPSE
  }

  public enum WeightInitializationType {
    ZEROS, PARAMS, RANDOM
  }

  public record State(
      double[][][] weights,
      double[][] activations
  ) {
  }
}
