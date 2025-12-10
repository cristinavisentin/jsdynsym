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
/*
 * Copyright 2025 eric
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.ericmedvet.jsdynsym.buildable.builders;

import io.github.ericmedvet.jnb.core.Cacheable;
import io.github.ericmedvet.jnb.core.Discoverable;
import io.github.ericmedvet.jnb.core.Param;
import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jsdynsym.core.composed.InStepped;
import io.github.ericmedvet.jsdynsym.core.composed.OutStepped;
import io.github.ericmedvet.jsdynsym.core.composed.Stepped;
import io.github.ericmedvet.jsdynsym.core.numerical.*;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.DelayedRecurrentNetwork;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.HebbianMultiLayerPerceptron;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.MultiLayerPerceptron;
import java.util.List;
import java.util.function.Function;
import java.util.random.RandomGenerator;

@Discoverable(prefixTemplate = "dynamicalSystem|dynSys|ds.num")
public class NumericalDynamicalSystems {

  private NumericalDynamicalSystems() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Function<NumericalDynamicalSystem<?>, DelayedRecurrentNetwork> drn(
      @Param(value = "timeRange", dNPM = "m.range(min=0;max=1)") DoubleRange timeRange,
      @Param(value = "innerNeuronsRatio", dD = 1d) double innerNeuronsRatio,
      @Param(value = "innerNeurons", dD = 0) int innerNeurons,
      @Param(value = "activationFunction", dS = "tanh") MultiLayerPerceptron.ActivationFunction activationFunction,
      @Param(value = "threshold", dD = 0.1d) double threshold,
      @Param(value = "timeResolution", dD = 0.16666d) double timeResolution
  ) {
    return eNds -> new DelayedRecurrentNetwork(
        activationFunction,
        eNds.nOfInputs(),
        eNds.nOfOutputs(),
        innerNeurons > 0 ? innerNeurons : ((int) Math.round(
            innerNeuronsRatio * (eNds.nOfInputs() + eNds.nOfOutputs())
        )),
        timeRange,
        threshold,
        timeResolution
    );
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <S> Function<NumericalDynamicalSystem<?>, EnhancedInput<S>> enhanced(
      @Param("windowT") double windowT,
      @Param("inner") Function<NumericalDynamicalSystem<?>, NumericalDynamicalSystem<S>> inner,
      @Param(
          value = "types", dSs = {"current", "trend", "avg"}) List<EnhancedInput.Type> types
  ) {
    return eNds -> new EnhancedInput<>(
        inner.apply(MultivariateRealFunction.from(eNds.nOfInputs(), eNds.nOfOutputs())),
        windowT,
        types
    );
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Function<NumericalDynamicalSystem<?>, HebbianMultiLayerPerceptron> hebbianMlp(
      @Param(value = "innerLayerRatio", dD = 0.65) double innerLayerRatio,
      @Param(value = "nOfInnerLayers", dI = 1) int nOfInnerLayers,
      @Param("innerLayers") List<Integer> innerLayers,
      @Param(value = "learningRate", dD = 0.01) double learningRate,
      @Param(value = "activationFunction", dS = "tanh") MultiLayerPerceptron.ActivationFunction activationFunction,
      @Param(value = "initialWeightRange", dNPM = "m.range(min=-0.1;max=0.1)") DoubleRange initialWeightRange,
      @Param(value = "randomGenerator", dNPM = "m.defaultRG()") RandomGenerator randomGenerator,
      @Param(value = "parametrizationType", dS = "synapse") HebbianMultiLayerPerceptron.ParametrizationType parametrizationType,
      @Param(value = "weightInitializationType", dS = "params") HebbianMultiLayerPerceptron.WeightInitializationType weightInitializationType
  ) {
    return eNds -> new HebbianMultiLayerPerceptron(
        activationFunction,
        eNds.nOfInputs(),
        innerLayers.stream().mapToInt(i -> i).toArray(),
        eNds.nOfOutputs(),
        learningRate,
        initialWeightRange,
        randomGenerator,
        parametrizationType,
        weightInitializationType
    );
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <S> Function<NumericalDynamicalSystem<?>, NumericalDynamicalSystem<Stepped.State<S>>> inStepped(
      @Param(value = "stepT", dD = 1) double interval,
      @Param("inner") Function<NumericalDynamicalSystem<?>, NumericalDynamicalSystem<S>> inner
  ) {
    return eNds -> NumericalDynamicalSystem.from(
        new InStepped<>(
            inner.apply(MultivariateRealFunction.from(eNds.nOfInputs(), eNds.nOfOutputs())),
            interval
        ),
        eNds.nOfInputs(),
        eNds.nOfOutputs()
    );
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Function<NumericalDynamicalSystem<?>, LinearCombination> linear(
      @Param("zeroQ") boolean zeroQ
  ) {
    return eNds -> new LinearCombination(
        eNds.nOfInputs(),
        eNds.nOfOutputs(),
        zeroQ
    );
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Function<NumericalDynamicalSystem<?>, MultiLayerPerceptron> mlp(
      @Param(value = "innerLayerRatio", dD = 0.65) double innerLayerRatio,
      @Param(value = "nOfInnerLayers", dI = 1) int nOfInnerLayers,
      @Param("innerLayers") List<Integer> innerLayers,
      @Param(value = "activationFunction", dS = "tanh") MultiLayerPerceptron.ActivationFunction activationFunction
  ) {
    return eNds -> {
      if (innerLayers.isEmpty()) {
        int[] innerNeurons = new int[nOfInnerLayers];
        int centerSize = (int) Math.max(2, Math.round(eNds.nOfInputs() * innerLayerRatio));
        if (nOfInnerLayers > 1) {
          for (int i = 0; i < nOfInnerLayers / 2; i++) {
            innerNeurons[i] = eNds.nOfInputs() + (centerSize - eNds.nOfInputs()) / (nOfInnerLayers / 2 + 1) * (i + 1);
          }
          for (int i = nOfInnerLayers / 2; i < nOfInnerLayers; i++) {
            innerNeurons[i] = centerSize + (eNds
                .nOfOutputs() - centerSize) / (nOfInnerLayers / 2 + 1) * (i - nOfInnerLayers / 2);
          }
        } else if (nOfInnerLayers > 0) {
          innerNeurons[0] = centerSize;
        }
        return new MultiLayerPerceptron(
            activationFunction,
            eNds.nOfInputs(),
            innerNeurons,
            eNds.nOfOutputs()
        );
      } else {
        return new MultiLayerPerceptron(
            activationFunction,
            eNds.nOfInputs(),
            innerLayers.stream().mapToInt(i -> i).toArray(),
            eNds.nOfOutputs()
        );
      }
    };
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <S> Function<NumericalDynamicalSystem<?>, Noised<S>> noised(
      @Param(value = "inputSigma", dD = 0.01) double inputSigma,
      @Param(value = "outputSigma", dD = 0.01) double outputSigma,
      @Param(value = "randomGenerator", dNPM = "m.defaultRG()") RandomGenerator randomGenerator,
      @Param("inner") Function<NumericalDynamicalSystem<?>, NumericalDynamicalSystem<S>> inner
  ) {
    return eNds -> new Noised<>(
        inner.apply(MultivariateRealFunction.from(eNds.nOfInputs(), eNds.nOfOutputs())),
        inputSigma,
        outputSigma,
        randomGenerator
    );
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <S> Function<NumericalDynamicalSystem<?>, NumericalDynamicalSystem<Stepped.State<S>>> outStepped(
      @Param(value = "stepT", dD = 1) double interval,
      @Param("inner") Function<NumericalDynamicalSystem<?>, NumericalDynamicalSystem<S>> inner
  ) {
    return eNds -> NumericalDynamicalSystem.from(
        new OutStepped<>(
            inner.apply(MultivariateRealFunction.from(eNds.nOfInputs(), eNds.nOfOutputs())),
            interval
        ),
        eNds.nOfInputs(),
        eNds.nOfOutputs()
    );
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Function<NumericalDynamicalSystem<?>, Sinusoidal> sin(
      @Param(value = "p", dNPM = "m.range(min=-1.57;max=1.57)") DoubleRange phaseRange,
      @Param(value = "f", dNPM = "m.range(min=0;max=1)") DoubleRange frequencyRange,
      @Param(value = "a", dNPM = "m.range(min=0;max=1)") DoubleRange amplitudeRange,
      @Param(value = "b", dNPM = "m.range(min=-0.5;max=0.5)") DoubleRange biasRange
  ) {
    return eNds -> new Sinusoidal(
        eNds.nOfInputs(),
        eNds.nOfOutputs(),
        phaseRange,
        frequencyRange,
        amplitudeRange,
        biasRange
    );
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <S> Function<NumericalDynamicalSystem<?>, NumericalDynamicalSystem<Stepped.State<S>>> stepped(
      @Param(value = "stepT", dD = 0.1) double interval,
      @Param("inner") Function<NumericalDynamicalSystem<?>, NumericalDynamicalSystem<S>> inner
  ) {
    return eNds -> NumericalDynamicalSystem.from(
        new Stepped<>(
            inner.apply(MultivariateRealFunction.from(eNds.nOfInputs(), eNds.nOfOutputs())),
            interval
        ),
        eNds.nOfInputs(),
        eNds.nOfOutputs()
    );
  }
}
