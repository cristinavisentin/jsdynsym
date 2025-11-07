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
import io.github.ericmedvet.jsdynsym.core.numerical.LinearAlgebraUtils;
import io.github.ericmedvet.jsdynsym.core.numerical.MultivariateRealFunction;
import io.github.ericmedvet.jsdynsym.core.numerical.NumericalStatelessSystem;
import io.github.ericmedvet.jsdynsym.core.rl.LinearActorCritic.State;
import java.util.Arrays;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

public class LinearActorCritic implements NumericalTimeInvariantReinforcementLearningAgent<State>, FrozenableNumericalRLAgent<State> {

  public record State(double[][] actorWeights, double[] criticWeights) {

  }

  // hyperparameters
  private final double actorLearningRate;
  private final double criticLearningRate;
  private final double actorWeightDecay;
  private final double criticWeightDecay;
  private final double discountFactor;
  private final double explorationNoise;
  private final DoubleRange gradLogProbRange;
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
      double actorWeightDecay,
      double criticWeightDecay,
      double discountFactor,
      double explorationNoise,
      double maxGradLogProb,
      DoubleRange initialWeightRange,
      RandomGenerator randomGenerator
  ) {
    this.actorLearningRate = actorLearningRate;
    this.criticLearningRate = criticLearningRate;
    this.actorWeightDecay = actorWeightDecay;
    this.criticWeightDecay = criticWeightDecay;
    this.discountFactor = discountFactor;
    this.explorationNoise = explorationNoise;
    gradLogProbRange = new DoubleRange(-maxGradLogProb, maxGradLogProb);
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
    if (Objects.nonNull(lastAction) && Objects.nonNull(lastObservation) && !Double.isNaN(reward)) {
      double vLast = LinearAlgebraUtils.dotProduct(state.criticWeights, lastObservation);
      double vCurrent = LinearAlgebraUtils.dotProduct(state.criticWeights, observation);
      double tdError = reward + discountFactor * vCurrent - vLast;
      for (int i = 0; i < state.criticWeights.length; i++) {
        double decay = criticLearningRate * criticWeightDecay * state.criticWeights[i];
        state.criticWeights[i] = state.criticWeights[i] + criticLearningRate * tdError * lastObservation[i] - decay;
      }
      double[] meanAction = LinearAlgebraUtils.product(state.actorWeights, observation);
      double invSigmaSq = 1d / (explorationNoise * explorationNoise);
      for (int j = 0; j < nOfOutputs; j++) {
        double actionDifference = lastAction[j] - meanAction[j];
        for (int i = 0; i < nOfInputs; i++) {
          double gradLogProb = gradLogProbRange.clip(actionDifference * invSigmaSq * lastObservation[i]);
          double decay = actorLearningRate * actorWeightDecay * state.actorWeights[j][i];
          state.actorWeights[j][i] = state.actorWeights[j][i] + actorLearningRate * tdError * gradLogProb - decay;
        }
      }
    }
    // compute action
    double[] action = LinearAlgebraUtils.product(state.actorWeights, observation);
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

  @Override
  public NumericalStatelessSystem dynamicalSystem() {
    double[][] frozenActorWeights = Arrays.stream(state.actorWeights)
        .map(v -> Arrays.copyOf(v, v.length))
        .toArray(double[][]::new);
    return MultivariateRealFunction.from(
        observation -> LinearAlgebraUtils.product(frozenActorWeights, observation),
        nOfInputs,
        nOfOutputs
    );
  }
}
