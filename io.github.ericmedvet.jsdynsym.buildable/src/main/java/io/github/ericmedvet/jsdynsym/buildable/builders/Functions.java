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
import io.github.ericmedvet.jnb.datastructure.FormattedNamedFunction;
import io.github.ericmedvet.jnb.datastructure.NamedFunction;
import io.github.ericmedvet.jsdynsym.control.HomogeneousBiSimulation;
import io.github.ericmedvet.jsdynsym.control.Simulation;
import io.github.ericmedvet.jsdynsym.control.Simulation.Outcome;
import io.github.ericmedvet.jsdynsym.control.SingleAgentTask;
import io.github.ericmedvet.jsdynsym.control.SingleAgentTask.Step;
import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;
import io.github.ericmedvet.jsdynsym.core.FrozenableDynamicalSystem;
import io.github.ericmedvet.jsdynsym.core.StatelessSystem;
import io.github.ericmedvet.jsdynsym.core.composed.Composed;
import io.github.ericmedvet.jsdynsym.core.numerical.ann.MultiLayerPerceptron;
import io.github.ericmedvet.jsdynsym.core.rl.FrozenableRLAgent;
import io.github.ericmedvet.jsdynsym.core.rl.ReinforcementLearningAgent.RewardedInput;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.function.Function;
import java.util.stream.Collectors;

@Discoverable(prefixTemplate = "dynamicalSystem|dynSys|ds.function|f")
public class Functions {

  private Functions() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> cumulatedReward(
      @Param(value = "of", dNPM = "f.identity()") Function<X, Outcome<SingleAgentTask.Step<RewardedInput<?>, ?, ?>>> beforeF,
      @Param(value = "format", dS = "%+6.3f") String format
  ) {
    Function<Outcome<Step<RewardedInput<?>, ?, ?>>, Double> f = o -> o.snapshots()
        .values()
        .stream()
        .mapToDouble(step -> step.observation().reward())
        .sum();
    return FormattedNamedFunction.from(f, format, "cumulated.reward").compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, Double> doubleOp(
      @Param(value = "of", dNPM = "f.identity()") Function<X, Double> beforeF,
      @Param(value = "activationF", dS = "identity") MultiLayerPerceptron.ActivationFunction activationF,
      @Param(value = "format", dS = "%.1f") String format
  ) {
    Function<Double, Double> f = activationF::applyAsDouble;
    return FormattedNamedFunction.from(f, format, activationF.name().toLowerCase())
        .compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, C> NamedFunction<X, C> inner(
      @Param(value = "of", dNPM = "f.identity()") Function<X, Composed<C>> beforeF
  ) {
    Function<Composed<C>, C> f = Composed::inner;
    return NamedFunction.from(f, "inner").compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, I, O> FormattedNamedFunction<X, DynamicalSystem<I, O, ?>> nonLearning(
      @Param(value = "name", dS = "non.learning") String name,
      @Param(value = "of", dNPM = "f.identity()") Function<X, FrozenableRLAgent<I, O, ?>> beforeF,
      @Param(value = "format", dS = "%s") String format
  ) {
    Function<FrozenableRLAgent<I, O, ?>, DynamicalSystem<I, O, ?>> f = FrozenableRLAgent::dynamicalSystem;
    return FormattedNamedFunction.from(f, format, name).compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, S, B extends Simulation.Outcome<SS>, SS> NamedFunction<X, Simulation.Outcome<SS>> opponentBiSimulator(
      @Param(value = "of", dNPM = "f.identity()") Function<X, S> beforeF,
      @Param("simulation") HomogeneousBiSimulation<S, SS, B> biSimulation,
      @Param("opponent") S opponent,
      @Param(value = "home", dB = true) boolean home,
      @Param("tRange") DoubleRange tRange,
      @Param("dT") double dT,
      @Param(value = "format", dS = "%s") String format
  ) {
    Function<S, Simulation.Outcome<SS>> f = s -> home ? biSimulation.simulate(
        s,
        opponent,
        dT,
        tRange
    ) : biSimulation
        .simulate(opponent, s, dT, tRange);
    return NamedFunction.from(f, "opponent.sim").compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, S, B extends Simulation.Outcome<SS>, SS> NamedFunction<X, Simulation.Outcome<SS>> selfBiSimulator(
      @Param(value = "of", dNPM = "f.identity()") Function<X, S> beforeF,
      @Param("simulation") HomogeneousBiSimulation<S, SS, B> biSimulation,
      @Param("tRange") DoubleRange tRange,
      @Param("dT") double dT,
      @Param(value = "format", dS = "%s") String format
  ) {
    Function<S, Simulation.Outcome<SS>> f = s -> biSimulation.simulate(s, s, dT, tRange);
    return NamedFunction.from(f, "self.sim").compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, S> NamedFunction<X, SortedMap<Double, S>> simOutcome(
      @Param(value = "of", dNPM = "f.identity()") Function<X, Simulation.Outcome<S>> beforeF,
      @Param(value = "format", dS = "%s") String format
  ) {
    Function<Simulation.Outcome<S>, SortedMap<Double, S>> f = Simulation.Outcome::snapshots;
    return NamedFunction.from(f, "sim.outcome").compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, SS, O extends Simulation.Outcome<SS>, S extends Simulation<T, SS, O>, T> Function<X, O> simulate(
      @Param(value = "of", dNPM = "f.identity()") Function<X, T> beforeF,
      @Param("simulation") S simulation,
      @Param("tRange") DoubleRange tRange,
      @Param("dT") double dT,
      @Param(value = "format", dS = "%s") String format
  ) {
    Function<T, O> f = t -> simulation.simulate(t, dT, tRange);
    return FormattedNamedFunction.from(f, format, "sim[%s]".formatted(simulation)).compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, I, O> FormattedNamedFunction<X, StatelessSystem<I, O>> stateless(
      @Param(value = "name", dS = "stateless") String name,
      @Param(value = "of", dNPM = "f.identity()") Function<X, FrozenableDynamicalSystem<I, O, ?>> beforeF,
      @Param(value = "format", dS = "%s") String format
  ) {
    Function<FrozenableDynamicalSystem<I, O, ?>, StatelessSystem<I, O>> f = FrozenableDynamicalSystem::stateless;
    return FormattedNamedFunction.from(f, format, name).compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, O, A, S> NamedFunction<X, Outcome<SingleAgentTask.Step<O, A, S>>> unwrappedRl(
      @Param(value = "of", dNPM = "f.identity()") Function<X, Outcome<SingleAgentTask.Step<RewardedInput<O>, A, S>>> beforeF,
      @Param(value = "name", dS = "unwrapped.rl") String name
  ) {
    Function<Outcome<SingleAgentTask.Step<RewardedInput<O>, A, S>>, Outcome<SingleAgentTask.Step<O, A, S>>> f = rlO -> Outcome
        .of(
            rlO.snapshots()
                .entrySet()
                .stream()
                .collect(
                    Collectors.toMap(
                        Entry::getKey,
                        e -> new Step<>(
                            e.getValue().observation().input(),
                            e.getValue().action(),
                            e.getValue().state()
                        ),
                        (s1, s2) -> s1,
                        TreeMap::new
                    )
                )
        );
    return NamedFunction.from(f, name).compose(beforeF);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X> FormattedNamedFunction<X, List<List<List<Double>>>> weights(
      @Param(value = "of", dNPM = "f.identity()") Function<X, MultiLayerPerceptron> beforeF,
      @Param(value = "format", dS = "%s") String format
  ) {
    Function<MultiLayerPerceptron, List<List<List<Double>>>> f = mlp -> {
      int[] neurons = new int[mlp.nOfLayers()];
      for (int i = 0; i < neurons.length; i++) {
        neurons[i] = mlp.sizeOfLayer(i);
      }
      double[][][] unflat = MultiLayerPerceptron.unflat(mlp.getParams(), neurons);
      return Arrays.stream(unflat)
          .map(
              layerWs -> Arrays.stream(layerWs)
                  .map(
                      neuronWs -> Arrays.stream(
                          neuronWs
                      ).boxed().toList()
                  )
                  .toList()
          )
          .toList();
    };
    return FormattedNamedFunction.from(f, format, "weights").compose(beforeF);
  }

}
