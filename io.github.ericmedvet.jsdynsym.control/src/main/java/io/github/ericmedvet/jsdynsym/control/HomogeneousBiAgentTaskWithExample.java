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
import java.util.function.Predicate;
import java.util.function.Supplier;

public interface HomogeneousBiAgentTaskWithExample<C extends DynamicalSystem<O, A, ?>, O, A, S>
    extends HomogeneousBiAgentTask<C, O, A, S>,
        HomogeneousBiSimulationWithExample<
            C,
            HomogeneousBiAgentTask.Step<O, A, S>,
            Simulation.Outcome<HomogeneousBiAgentTask.Step<O, A, S>>> {
  C example();

  static <C extends DynamicalSystem<O, A, ?>, O, A, S> HomogeneousBiAgentTask<C, O, A, S> fromHomogenousBiEnvironment(
      Supplier<HomogeneousBiEnvironmentWithExample<O, A, S>> biEnvironmentSupplier,
      Pair<A, A> initialActions,
      Predicate<S> stopCondition,
      DoubleRange tRange,
      double dT) {
    HomogeneousBiAgentTask<C, O, A, S> hbe = HomogeneousBiAgentTask.fromHomogenousBiEnvironment(
        biEnvironmentSupplier::get, initialActions, stopCondition, tRange, dT);
    return new HomogeneousBiAgentTaskWithExample<>() {
      @Override
      public C example() {
        return (C) biEnvironmentSupplier.get().example(); // TODO check
      }

      @Override
      public Outcome<Step<O, A, S>> simulate(C c1, C c2) {
        return hbe.simulate(c1, c2);
      }
    };
  }

  static <C extends DynamicalSystem<O, A, ?>, O, A, S> HomogeneousBiAgentTask<C, O, A, S> fromHomogenousBiEnvironment(
      Supplier<HomogeneousBiEnvironmentWithExample<O, A, S>> biEnvironmentSupplier,
      Predicate<S> stopCondition,
      DoubleRange tRange,
      double dT) {
    return HomogeneousBiAgentTaskWithExample.fromHomogenousBiEnvironment(
        biEnvironmentSupplier, biEnvironmentSupplier.get().defaultActions(), stopCondition, tRange, dT);
  }
}
