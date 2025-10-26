/*-
 * ========================LICENSE_START=================================
 * jsdynsym-buildable
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
package io.github.ericmedvet.jsdynsym.buildable.builders;

import io.github.ericmedvet.jnb.core.Cacheable;
import io.github.ericmedvet.jnb.core.Discoverable;
import io.github.ericmedvet.jnb.core.Param;
import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jsdynsym.core.rl.LinearActorCritic;
import io.github.ericmedvet.jsdynsym.core.rl.NumericalReinforcementLearningAgent;
import java.util.function.UnaryOperator;
import java.util.random.RandomGenerator;

@Discoverable(prefixTemplate = "dynamicalSystem|dynSys|ds.rl")
public class RLAgents {

  private RLAgents() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static UnaryOperator<NumericalReinforcementLearningAgent<LinearActorCritic.State>> linearActorCritic(
      @Param(value = "name", iS = "lAC[alr={actorLearningRate};clr={criticLearningRate};en={explorationNoise}]") String name,
      @Param(value = "actorLearningRate", dD = 0.0001) double actorLearningRate,
      @Param(value = "criticLearningRate", dD = 0.001) double criticLearningRate,
      @Param(value = "actorWeightDecay", dD = 0.00001) double actorWeightDecay,
      @Param(value = "criticWeightDecay", dD = 0.0001) double criticWeightDecay,
      @Param(value = "discountFactor", dD = 0.99) double discountFactor,
      @Param(value = "explorationNoise", dD = 0.1) double explorationNoise,
      @Param(value = "maxGradLogProb", dD = 10) double maxGradLogProb,
      @Param(value = "initialWeightRange", dNPM = "m.range(min=-0.2;max=0.2)") DoubleRange initialWeightRange,
      @Param(value = "randomGenerator", dNPM = "m.defaultRG()") RandomGenerator randomGenerator
  ) {
    return exampleAgent -> new LinearActorCritic(
        exampleAgent.nOfInputs(),
        exampleAgent.nOfOutputs(),
        actorLearningRate,
        criticLearningRate,
        actorWeightDecay,
        criticWeightDecay,
        discountFactor,
        explorationNoise,
        maxGradLogProb,
        initialWeightRange,
        randomGenerator
    );
  }
}
