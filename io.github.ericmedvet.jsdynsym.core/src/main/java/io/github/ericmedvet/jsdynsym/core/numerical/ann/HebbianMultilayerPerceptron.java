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

import io.github.ericmedvet.jnb.datastructure.NumericalParametrized;
import io.github.ericmedvet.jsdynsym.core.numerical.NumericalTimeInvariantDynamicalSystem;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class HebbianMultilayerPerceptron implements NumericalTimeInvariantDynamicalSystem<HebbianMultilayerPerceptron.State>, NumericalParametrized<HebbianMultilayerPerceptron> {
  private final MultiLayerPerceptron.ActivationFunction activationFunction;
  private final double[][][] as;
  private final double[][][] bs;
  private final double[][][] cs;
  private final double[][][] ds;
  private final double[][][] initialWeights;
  private final int[] neurons;
  private final double learningRate;
  private final ParametrizationType parametrizationType;
  private final WeightInitializationType weightInitializationType;

  public record State(
      double[][][] weights,
      double[][] activations
  ) {
  }

  public enum ParametrizationType {
    NETWORK, LAYER, NEURON, SYNAPSE
  }

  public enum WeightInitializationType {
    ZEROS, PARAMS, RANDOM
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
      case NETWORK -> MultiLayerPerceptron.unflat(nCopies(params.length, params[0]), neurons);
      case LAYER -> IntStream.range(1, neurons.length)
          .mapToObj(
              li -> IntStream.range(0, neurons[li])
                  .mapToObj(ni -> nCopies(neurons[li - 1] + 1, params[li - 1]))
                  .toArray(double[][]::new)
          )
          .toArray(double[][][]::new);
      case NEURON -> neuronUnflat(params, neurons);
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

  private static double[] concat(double[] vs1, double[] vs2) {
    double[] vs = new double[vs1.length + vs2.length];
    System.arraycopy(vs1, 0, vs, 0, vs1.length);
    System.arraycopy(vs2, 0, vs, vs1.length, vs2.length);
    return vs;
  }

  private static double[] nCopies(int n, double value) {
    double[] values = new double[n];
    Arrays.fill(values, value);
    return values;
  }

  private static double[] flat(ParametrizationType parametrizationType, double[][][] params, int[] neurons) {
    return switch (parametrizationType) {
      case NETWORK -> new double[]{params[0][0][0]};
      case LAYER -> Arrays.stream(params).mapToDouble(l -> l[0][0]).toArray();
      case NEURON ->
        Arrays.stream(params)
            .flatMap(l -> Arrays.stream(l).mapToDouble(n -> n[0]).boxed())
            .mapToDouble(v -> v)
            .toArray();
      case SYNAPSE -> MultiLayerPerceptron.flat(params, neurons);
    };
  }

  private State state;

  public HebbianMultilayerPerceptron(
      MultiLayerPerceptron.ActivationFunction activationFunction,
      double[][][] as,
      double[][][] bs,
      double[][][] cs,
      double[][][] ds,
      double[][][] initialWeights,
      int[] neurons,
      double learningRate,
      ParametrizationType parametrizationType,
      WeightInitializationType weightInitializationType
  ) {
    this.activationFunction = activationFunction;
    this.as = as;
    this.bs = bs;
    this.cs = cs;
    this.ds = ds;
    this.initialWeights = initialWeights;
    this.neurons = neurons;
    this.learningRate = learningRate;
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
      ParametrizationType parametrizationType,
      WeightInitializationType weightInitializationType
  ) {
    this(
        activationFunction,
        unflat(
            params,
            0,
            MultiLayerPerceptron.countWeights(MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput)),
            MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput)
        ),
        unflat(
            params,
            MultiLayerPerceptron.countWeights(MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput)),
            2 * MultiLayerPerceptron.countWeights(MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput)),
            MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput)
        ),
        unflat(
            params,
            2 * MultiLayerPerceptron.countWeights(MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput)),
            3 * MultiLayerPerceptron.countWeights(MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput)),
            MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput)
        ),
        unflat(
            params,
            3 * MultiLayerPerceptron.countWeights(MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput)),
            4 * MultiLayerPerceptron.countWeights(MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput)),
            MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput)
        ),
        unflat(
            params,
            4 * MultiLayerPerceptron.countWeights(MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput)),
            params.length,
            MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput)
        ),
        MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput),
        learningRate,
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
      ParametrizationType parametrizationType,
      WeightInitializationType weightInitializationType
  ) {
    this(
        activationFunction,
        nOfInput,
        innerNeurons,
        nOfOutput,
        new double[nOfParams(MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput))],
        learningRate,
        parametrizationType,
        weightInitializationType
    );
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

  public static int nOfParams(int[] neurons) {
    return 5 * MultiLayerPerceptron.countWeights(neurons);
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
    state = new State(
        initialWeights,
        Arrays.stream(neurons).mapToObj(double[]::new).toArray(double[][]::new)
    );
  }

  @Override
  public double[] getParams() { // TODO re write me
    int nOfParams = nOfParams(neurons);
    double[] flatParams = new double[5 * nOfParams];
    double[] flatWeights = MultiLayerPerceptron.flat(state.weights, neurons);
    double[] flatAs = MultiLayerPerceptron.flat(as, neurons);
    double[] flatBs = MultiLayerPerceptron.flat(bs, neurons);
    double[] flatCs = MultiLayerPerceptron.flat(cs, neurons);
    double[] flatDs = MultiLayerPerceptron.flat(ds, neurons);
    System.arraycopy(flatAs, 0, flatParams, 0, flatAs.length);
    System.arraycopy(flatBs, 0, flatParams, nOfParams, flatBs.length);
    System.arraycopy(flatCs, 0, flatParams, 2 * nOfParams, flatCs.length);
    System.arraycopy(flatDs, 0, flatParams, 3 * nOfParams, flatDs.length);
    System.arraycopy(flatWeights, 0, flatParams, 4 * nOfParams, flatWeights.length);
    return flatParams;
  }

  private static double[][][] unflat(double[] params, int startIdx, int stopIdx, int[] neurons) { // TODO remove me
    return MultiLayerPerceptron.unflat(Arrays.copyOfRange(params, startIdx, stopIdx), neurons);
  }

  private static void set(double[][][] src, double[][][] dst) {
    for (int l = 0; l < src.length; l++) {
      for (int s = 0; s < src[l].length; s++) {
        System.arraycopy(src[l][s], 0, dst[l][s], 0, src[l][s].length);
      }
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
}
