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
import io.github.ericmedvet.jsdynsym.control.SingleAgentTask.Step;
import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import java.util.function.Predicate;
import java.util.function.Supplier;

public interface SingleAgentTask<C extends DynamicalSystem<O, A, ?>, O, A, S> extends Simulation<C, Step<O, A, S>, Simulation.Outcome<Step<O, A, S>>> {

  record Step<O, A, S>(O observation, A action, S state) {}

  static <C extends DynamicalSystem<O, A, ?>, O, A, S> SingleAgentTask<C, O, A, S> fromEnvironment(
      Supplier<? extends DynamicalSystem<A, O, S>> environmentSupplier,
      O initialObservation,
      C exampleAgent,
      Predicate<S> stopCondition,
      boolean resetAgent
  ) {
    return new SingleAgentTask<>() {
      @Override
      public Outcome<Step<O, A, S>> simulate(C agent, double dT, DoubleRange tRange) {
        DynamicalSystem<A, O, S> environment = environmentSupplier.get();
        environment.reset();
        if (resetAgent) {
          agent.reset();
        }
        double t = tRange.min();
        Map<Double, Step<O, A, S>> steps = new HashMap<>();
        O observation = initialObservation;
        while (t <= tRange.max() && !stopCondition.test(environment.getState())) {
          A action = agent.step(t, observation);
          observation = environment.step(t, action);
          steps.put(t, new Step<>(observation, action, environment.getState()));
          t = t + dT;
        }
        return Outcome.of(new TreeMap<>(steps));
      }

      @Override
      public Optional<C> example() {
        return Optional.of(exampleAgent);
      }
    };
  }

  static <C extends DynamicalSystem<O, A, ?>, O, A, S> SingleAgentTask<C, O, A, S> fromEnvironment(
      Supplier<Environment<O, A, S, C>> environmentSupplier,
      Predicate<S> stopCondition,
      boolean resetAgent
  ) {
    return fromEnvironment(
        environmentSupplier,
        environmentSupplier.get().defaultObservation(),
        environmentSupplier.get().exampleAgent(),
        stopCondition,
        resetAgent
    );
  }

}
