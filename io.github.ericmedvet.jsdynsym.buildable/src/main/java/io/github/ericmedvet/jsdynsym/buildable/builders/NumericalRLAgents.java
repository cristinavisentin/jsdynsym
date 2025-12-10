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
import io.github.ericmedvet.jsdynsym.core.numerical.NumericalDynamicalSystem;
import io.github.ericmedvet.jsdynsym.core.numerical.UnivariateRealFunction;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.HebbianMultiLayerPerceptron;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.MultiLayerPerceptron;
import io.github.ericmedvet.jsdynsym.core.numerical.named.NamedUnivariateRealFunction;
import io.github.ericmedvet.jsdynsym.core.rl.FreeFormPlasticMLPRLAgent;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.random.RandomGenerator;

@Discoverable(prefixTemplate = "dynamicalSystem|dynSys|ds.nrl") // TODO choose prefix template
public class NumericalRLAgents {
  private NumericalRLAgents() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Function<NumericalDynamicalSystem<?>, FreeFormPlasticMLPRLAgent> freeFormMlp(
      @Param(value = "innerLayerRatio", dD = 0.65) double innerLayerRatio,
      @Param(value = "nOfInnerLayers", dI = 1) int nOfInnerLayers,
      @Param("innerLayers") List<Integer> innerLayers,
      @Param(value = "activationFunction", dS = "tanh") MultiLayerPerceptron.ActivationFunction activationFunction,
      @Param(value = "historyLength", dI = 10) int historyLength,
      @Param(value = "initialWeightRange", dNPM = "m.range(min=-0.1;max=0.1)") DoubleRange initialWeightRange,
      @Param(value = "randomGenerator", dNPM = "m.defaultRG()") RandomGenerator randomGenerator,
      @Param(value = "weightInitializationType", dS = "random") HebbianMultiLayerPerceptron.WeightInitializationType weightInitializationType
  ) {
    List<String> variableNames = FreeFormPlasticMLPRLAgent.getVariableNames();
    NamedUnivariateRealFunction plasticityFunction = NamedUnivariateRealFunction.from(
        UnivariateRealFunction.from(
            inputs -> 0 + Arrays.stream(inputs).sum(),
            variableNames.size()
        ),
        variableNames,
        "Output"
    );
    return eNds -> new FreeFormPlasticMLPRLAgent(
        activationFunction,
        plasticityFunction,
        eNds.nOfInputs(),
        innerLayers.stream().mapToInt(i -> i).toArray(),
        eNds.nOfOutputs(),
        historyLength,
        weightInitializationType,
        initialWeightRange,
        randomGenerator
    );
  }
}
