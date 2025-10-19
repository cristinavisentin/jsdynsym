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
import io.github.ericmedvet.jsdynsym.core.rl.LinearActorCritic.State;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

public class LinearActorCritic implements NumericalTimeInvariantReinforcementLearningAgent<State> {

  public record State(double[][] actorWeights, double[] criticWeights) {
  }

  // hyperparameters
  private final double actorLearningRate;
  private final double criticLearningRate;
  private final double discountFactor;
  private final double explorationNoise;
  private final DoubleRange initialWeightRange;
  private final int nOfInputs;
  private final int nOfOutputs;
  private final RandomGenerator randomGenerator;

  // state
  private State state;
  private double[] lastObservation;
  private double[] lastAction;

  public LinearActorCritic(
      int nOfInputs,
      int nOfOutputs,
      double actorLearningRate,
      double criticLearningRate,
      double discountFactor,
      double explorationNoise,
      DoubleRange initialWeightRange,
      RandomGenerator randomGenerator
  ) {
    this.actorLearningRate = actorLearningRate;
    this.criticLearningRate = criticLearningRate;
    this.discountFactor = discountFactor;
    this.explorationNoise = explorationNoise;
    this.initialWeightRange = initialWeightRange;
    this.nOfInputs = nOfInputs;
    this.nOfOutputs = nOfOutputs;
    this.randomGenerator = randomGenerator;
    reset();
  }

  @Override
  public int nOfInputs() {
    return nOfInputs;
  }

  @Override
  public int nOfOutputs() {
    return nOfOutputs;
  }

  @Override
  public double[] step(double[] observation, double reward) {
    if (observation.length != nOfInputs) {
      throw new IllegalArgumentException(
          String.format("Expected input length is %d: found %d", nOfInputs, observation.length)
      );
    }
    // learn
    if (Objects.nonNull(lastAction) && Objects.nonNull(lastObservation)) {
      double vLast = dotProduct(state.criticWeights, lastObservation);
      double vCurrent = dotProduct(state.criticWeights, observation);
      double tdError = reward + discountFactor * vCurrent - vLast;
      for (int i = 0; i < state.criticWeights.length; i++) {
        state.criticWeights[i] = state.criticWeights[i] + criticLearningRate * tdError * lastObservation[i];
      }
      double[] meanAction = product(state.actorWeights, observation);
      double invSigmaSq = 1d / (explorationNoise * explorationNoise);
      for (int j = 0; j < nOfOutputs; j++) {
        double actionDifference = lastAction[j] - meanAction[j];
        for (int i = 0; i < nOfInputs; i++) {
          double gradLogProb = actionDifference * invSigmaSq * lastObservation[i];
          state.actorWeights[j][i] = state.actorWeights[j][i] + actorLearningRate * tdError * gradLogProb;
        }
      }
    }
    // compute action
    double[] action = product(state.actorWeights, observation);
    for (int i = 0; i < action.length; i++) {
      action[i] = action[i] + randomGenerator.nextGaussian() * explorationNoise;
    }
    lastObservation = observation;
    lastAction = action;
    return action;
  }

  @Override
  public State getState() {
    return state;
  }

  @Override
  public void reset() {
    state = new State(
        IntStream.range(0, nOfOutputs)
            .mapToObj(
                iO -> IntStream.range(0, nOfInputs)
                    .mapToDouble(iI -> initialWeightRange.denormalize(randomGenerator.nextDouble()))
                    .toArray()
            )
            .toArray(double[][]::new),
        IntStream.range(0, nOfInputs)
            .mapToDouble(iI -> initialWeightRange.denormalize(randomGenerator.nextDouble()))
            .toArray()
    );
    lastObservation = null;
    lastAction = null;
  }

  private static double dotProduct(double[] v1, double[] v2) {
    if (v1.length != v2.length) {
      throw new IllegalArgumentException("Vectors must have the same length for dot product.");
    }
    double sum = 0d;
    for (int i = 0; i < v1.length; i++) {
      sum += v1[i] * v2[i];
    }
    return sum;
  }

  private static double[] product(double[][] m, double[] v) {
    double[] o = new double[m.length];
    for (int j = 0; j < o.length; j++) {
      o[j] = dotProduct(m[j], v);
    }
    return o;
  }
}
