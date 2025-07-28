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

import io.github.ericmedvet.jnb.datastructure.Pair;
import java.util.Optional;

public interface BiSimulation<T1, T2, S, O extends Simulation.Outcome<S>> extends Simulation<Pair<T1, T2>, S, O> {

  O simulate(T1 t1, T2 t2);

  @Override
  default O simulate(Pair<T1, T2> tPair) {
    return simulate(tPair.first(), tPair.second());
  }

  default Optional<T1> example1() {
    return Optional.empty();
  }

  default Optional<T2> example2() {
    return Optional.empty();
  }

  @Override
  default Optional<Pair<T1, T2>> example() {
    if (example1().isPresent() || example2().isPresent()) {
      return Optional.of(new Pair<>(example1().orElse(null), example2().orElse(null)));
    }
    return Optional.empty();
  }
}
