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

public class HebbianMultilayerPerceptron implements NumericalTimeInvariantDynamicalSystem<HebbianMultilayerPerceptron.State>, NumericalParametrized<HebbianMultilayerPerceptron> {
  private final MultiLayerPerceptron.ActivationFunction activationFunction;
  private final double[][][] as;
  private final double[][][] bs;
  private final double[][][] cs;
  private final double[][][] ds;
  private final double[][][] initialWeights;
  private final int[] neurons;
  private final double learningRate;

  public record State(
      double[][][] weights,
      double[][] activations
  ) {
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
      double learningRate
  ) {
    this.activationFunction = activationFunction;
    this.as = as;
    this.bs = bs;
    this.cs = cs;
    this.ds = ds;
    this.initialWeights = initialWeights;
    this.neurons = neurons;
    this.learningRate = learningRate;
    reset();
  }

  public HebbianMultilayerPerceptron(
      MultiLayerPerceptron.ActivationFunction activationFunction,
      int nOfInput,
      int[] innerNeurons,
      int nOfOutput,
      double[] params,
      double learningRate
  ) {
    this(
        activationFunction,
        unflat(params, MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput), ParamType.AS),
        unflat(params, MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput), ParamType.BS),
        unflat(params, MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput), ParamType.CS),
        unflat(params, MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput), ParamType.DS),
        unflat(params, MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput), ParamType.WEIGHTS),
        MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput),
        learningRate
    );
  }

  public HebbianMultilayerPerceptron(
      MultiLayerPerceptron.ActivationFunction activationFunction,
      int nOfInput,
      int[] innerNeurons,
      int nOfOutput,
      double learningRate
  ) {
    this(
        activationFunction,
        nOfInput,
        innerNeurons,
        nOfOutput,
        new double[nOfParams(MultiLayerPerceptron.countNeurons(nOfInput, innerNeurons, nOfOutput))],
        learningRate
    );
  }

  public enum ParamType {
    WEIGHTS, AS, BS, CS, DS
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

  public static double[] flat(
      double[][][] unflatWeights,
      double[][][] unflatAs,
      double[][][] unflatBs,
      double[][][] unflatCs,
      double[][][] unflatDs,
      int[] neurons
  ) {
    double[] flatParams = new double[nOfParams(neurons)];
    int c = 0;
    for (double[][][] param : new double[][][][]{unflatWeights, unflatAs, unflatBs, unflatCs, unflatDs}) {
      for (int l = 0; l < param.length; l++) {
        for (int j = 0; j < param[l].length; j++) {
          for (int k = 0; k < param[l][j].length; k++) {
            flatParams[c] = param[l][j][k];
            c += 1;
          }
        }
      }
    }
    return flatParams;
  }

  public static double[][][] unflat(double[] flatParams, int[] neurons, ParamType paramType) {
    int nParams = MultiLayerPerceptron.countWeights(neurons);
    double[][][] unflatParams = new double[neurons.length - 1][][];
    int startIndex, stopIndex;
    switch (paramType) {
      case WEIGHTS -> {
        startIndex = 0;
        stopIndex = nParams;
      }
      case AS -> {
        startIndex = nParams;
        stopIndex = 2 * nParams;
      }
      case BS -> {
        startIndex = 2 * nParams;
        stopIndex = 3 * nParams;
      }
      case CS -> {
        startIndex = 3 * nParams;
        stopIndex = 4 * nParams;
      }
      case DS -> {
        startIndex = 4 * nParams;
        stopIndex = flatParams.length;
      }
      default -> throw new IllegalArgumentException("Invalid paramType: " + paramType);
    }
    double[] params = Arrays.copyOfRange(flatParams, startIndex, stopIndex);
    int c = 0;
    for (int i = 1; i < neurons.length; i++) {
      unflatParams[i - 1] = new double[neurons[i]][neurons[i - 1] + 1];
      for (int j = 0; j < neurons[i]; j++) {
        for (int k = 0; k < neurons[i - 1] + 1; k++) {
          unflatParams[i - 1][j][k] = params[c];
          c += 1;
        }
      }
    }
    return unflatParams;
  }

  @Override
  public int nOfInputs() {
    return neurons[0];
  }

  @Override
  public int nOfOutputs() {
    return neurons[neurons.length - 1];
  }

  public static int nOfParams(int[] neurons) {
    return 5 * MultiLayerPerceptron.countWeights(neurons);
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
  public double[] getParams() {
    return flat(state.weights, as, bs, cs, ds, neurons);
  }

  @Override
  public void setParams(double[] params) {
    double[][][] newWeights = unflat(params, neurons, ParamType.WEIGHTS);
    double[][][] newAs = unflat(params, neurons, ParamType.AS);
    double[][][] newBs = unflat(params, neurons, ParamType.BS);
    double[][][] newCs = unflat(params, neurons, ParamType.CS);
    double[][][] newDs = unflat(params, neurons, ParamType.DS);
    for (int l = 0; l < newWeights.length; l++) {
      for (int s = 0; s < newWeights[l].length; s++) {
        System.arraycopy(newWeights[l][s], 0, state.weights[l][s], 0, newWeights[l][s].length);
        System.arraycopy(newAs[l][s], 0, this.as[l][s], 0, newAs[l][s].length);
        System.arraycopy(newBs[l][s], 0, this.bs[l][s], 0, newBs[l][s].length);
        System.arraycopy(newCs[l][s], 0, this.cs[l][s], 0, newCs[l][s].length);
        System.arraycopy(newDs[l][s], 0, this.ds[l][s], 0, newDs[l][s].length);
      }
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
