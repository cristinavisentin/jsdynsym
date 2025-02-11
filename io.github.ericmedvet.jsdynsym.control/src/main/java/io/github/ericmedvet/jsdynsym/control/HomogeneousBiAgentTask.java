/*-
 * ========================LICENSE_START=================================
 * jsdynsym-control
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
package io.github.ericmedvet.jsdynsym.control;

import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.Pair;
import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Predicate;
import java.util.function.Supplier;

public interface HomogeneousBiAgentTask<C extends DynamicalSystem<O, A, ?>, O, A, S>
    extends HomogeneousBiSimulation<
        C, HomogeneousBiAgentTask.Step<O, A, S>, Simulation.Outcome<HomogeneousBiAgentTask.Step<O, A, S>>> {

  record Step<O, A, S>(Pair<O, O> observations, Pair<A, A> actions, S state) {}

  static <C extends DynamicalSystem<O, A, ?>, O, A, S> HomogeneousBiAgentTask<C, O, A, S> fromHomogenousBiEnvironment(
      Supplier<HomogeneousBiEnvironment<O, A, S>> biEnvironmentSupplier,
      Pair<A, A> initialActions,
      Predicate<S> stopCondition,
      DoubleRange tRange,
      double dT) {
    return (agent1, agent2) -> {
      HomogeneousBiEnvironment<O, A, S> biEnvironment = biEnvironmentSupplier.get();
      biEnvironment.reset();
      agent1.reset();
      agent2.reset();
      double t = tRange.min();
      Map<Double, HomogeneousBiAgentTask.Step<O, A, S>> steps = new HashMap<>();
      Pair<O, O> observations = biEnvironment.step(t, initialActions);
      while (t <= tRange.max() && !stopCondition.test(biEnvironment.getState())) {
        Pair<A, A> actions =
            new Pair<>(agent1.step(t, observations.first()), agent2.step(t, observations.second()));
        observations = biEnvironment.step(t, actions);
        steps.put(t, new Step<>(observations, actions, biEnvironment.getState()));
        t = t + dT;
      }
      return Outcome.of(new TreeMap<>(steps));
    };
  }

  static <C extends DynamicalSystem<O, A, ?>, O, A, S> HomogeneousBiAgentTask<C, O, A, S> fromHomogenousBiEnvironment(
      Supplier<HomogeneousBiEnvironment<O, A, S>> biEnvironmentSupplier,
      Predicate<S> stopCondition,
      DoubleRange tRange,
      double dT) {
    return fromHomogenousBiEnvironment(
        biEnvironmentSupplier, biEnvironmentSupplier.get().defaultActions(), stopCondition, tRange, dT);
  }
}
