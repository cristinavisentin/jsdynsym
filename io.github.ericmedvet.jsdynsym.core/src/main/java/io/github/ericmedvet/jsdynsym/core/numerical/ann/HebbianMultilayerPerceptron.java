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

  @Override
  public double[] step(double[] input) {
    if (input.length != neurons[0]) {
      throw new IllegalArgumentException(
          String.format("Expected input length is %d: found %d", neurons[0], input.length)
      );
    }
    // update weights
    double[][][] newWeights = state.weights;
    for (int i = 1; i < newWeights.length + 1; i++) {
      for (int j = 0; j < newWeights[i - 1].length; j++) {
        double postActivation = state.activations[i][j];
        for (int k = 1; k < newWeights[i - 1][j].length; k++) {
          double preActivation = state.activations[i - 1][k - 1];
          newWeights[i - 1][j][k] += learningRate * (as[i - 1][j][k] * preActivation + bs[i - 1][j][k] * postActivation + cs[i - 1][j][k] * preActivation * postActivation + ds[i - 1][j][k]);
        }
      }
    }

    /*    for (int i = 1; i < neurons.length; i++) {
      for (int j = 0; j < neurons[i]; j++) {
        double postActivation = state.activations[i][j];
        for (int k = 1; k < neurons[i - 1] + 1; k++) {
          double preActivation = state.activations[i - 1][k];
          newWeights[i][j][k] += learningRate * (as[i][j][k] * preActivation + bs[i][j][k] * postActivation + cs[i][j][k] * preActivation * postActivation + ds[i][j][k]);
        }
      }
    }*/

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
    System.out.println(Arrays.deepToString(state.weights));
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
    state = new State(
        initialWeights,
        Arrays.stream(neurons).mapToObj(double[]::new).toArray(double[][]::new)
    );

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
  public double[] getParams() {
    return MultiLayerPerceptron.flat(state.weights, neurons);
  }

  @Override
  public void setParams(double[] params) {
    double[][][] newWeights = MultiLayerPerceptron.unflat(params, neurons);
    for (int l = 0; l < newWeights.length; l++) {
      for (int s = 0; s < newWeights[l].length; s++) {
        System.arraycopy(newWeights[l][s], 0, state.weights[l][s], 0, newWeights[l][s].length);
      }
    }
  }


}
