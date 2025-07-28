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

import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;

public interface HomogeneousBiEnvironment<O, A, S, C extends DynamicalSystem<O, A, ?>> extends BiEnvironment<O, O, A, A, S, C, C> {

  C exampleAgent();

  O defaultObservation();

  @Override
  default C exampleAgent1() {
    return exampleAgent();
  }

  @Override
  default C exampleAgent2() {
    return exampleAgent();
  }

  @Override
  default O defaultObservation1() {
    return defaultObservation();
  }

  @Override
  default O defaultObservation2() {
    return defaultObservation();
  }
}
