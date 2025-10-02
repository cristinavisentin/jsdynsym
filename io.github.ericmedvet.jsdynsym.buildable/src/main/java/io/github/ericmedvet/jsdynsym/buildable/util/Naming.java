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

package io.github.ericmedvet.jsdynsym.buildable.util;

import io.github.ericmedvet.jsdynsym.control.Environment;
import io.github.ericmedvet.jsdynsym.control.Simulation;
import io.github.ericmedvet.jsdynsym.control.SingleAgentTask;
import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;
import java.util.Optional;

public class Naming {

  private Naming() {
  }

  public static <O, A, S, C extends DynamicalSystem<O, A, ?>> Environment<O, A, S, C> named(
      String name,
      Environment<O, A, S, C> environment
  ) {
    return new Environment<>() {
      @Override
      public O defaultObservation() {
        return environment.defaultObservation();
      }

      @Override
      public C exampleAgent() {
        return environment.exampleAgent();
      }

      @Override
      public S getState() {
        return environment.getState();
      }

      @Override
      public void reset() {
        environment.reset();
      }

      @Override
      public O step(double t, A input) {
        return environment.step(t, input);
      }

      @Override
      public String toString() {
        return name;
      }
    };
  }

  public static <T, S, O extends Simulation.Outcome<S>> Simulation<T, S, O> named(
      String name,
      Simulation<T, S, O> simulation
  ) {
    return new Simulation<>() {
      @Override
      public Optional<T> example() {
        return simulation.example();
      }

      @Override
      public O simulate(T t) {
        return simulation.simulate(t);
      }

      @Override
      public String toString() {
        return name;
      }
    };
  }

  public static <C extends DynamicalSystem<O, A, ?>, O, A, S> SingleAgentTask<C, O, A, S> named(
      String name,
      SingleAgentTask<C, O, A, S> singleAgentTask
  ) {
    return new SingleAgentTask<>() {
      @Override
      public Optional<C> example() {
        return singleAgentTask.example();
      }

      @Override
      public Outcome<Step<O, A, S>> simulate(C c) {
        return singleAgentTask.simulate(c);
      }

      @Override
      public String toString() {
        return name;
      }
    };
  }

}
