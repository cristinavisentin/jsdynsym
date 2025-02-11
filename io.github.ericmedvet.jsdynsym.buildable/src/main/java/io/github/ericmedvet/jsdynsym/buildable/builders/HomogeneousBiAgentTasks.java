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

import io.github.ericmedvet.jnb.core.*;
import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jsdynsym.control.*;
import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;
import java.util.function.Predicate;
import java.util.function.Supplier;

@Discoverable(prefixTemplate = "dynamicalSystem|dynSys|ds.biAgentTask|baTask|bat")
public class HomogeneousBiAgentTasks {

  private HomogeneousBiAgentTasks() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <C extends DynamicalSystem<O, A, ?>, O, A, S> HomogeneousBiAgentTask<C, O, A, S> fromEnvironment(
      @Param(value = "name", iS = "{environment.name}") String name,
      @Param("environment") HomogeneousBiEnvironment<O, A, S> environment,
      @Param("stopCondition") Predicate<S> stopCondition,
      @Param("tRange") DoubleRange tRange,
      @Param("dT") double dT,
      @Param(value = "", injection = Param.Injection.BUILDER) NamedBuilder<?> nb,
      @Param(value = "", injection = Param.Injection.MAP) ParamMap map
  ) {
    if (environment instanceof HomogeneousBiEnvironmentWithExample<O, A, S> hbewe) {
      @SuppressWarnings("unchecked") Supplier<HomogeneousBiEnvironmentWithExample<O, A, S>> supplier = () -> (HomogeneousBiEnvironmentWithExample<O, A, S>) nb
          .build((NamedParamMap) map.value("environment", ParamMap.Type.NAMED_PARAM_MAP));
      return HomogeneousBiAgentTaskWithExample.fromHomogenousBiEnvironment(supplier, stopCondition, tRange, dT);
    }
    @SuppressWarnings("unchecked") Supplier<HomogeneousBiEnvironment<O, A, S>> supplier = () -> (HomogeneousBiEnvironment<O, A, S>) nb
        .build((NamedParamMap) map.value("environment", ParamMap.Type.NAMED_PARAM_MAP));
    return HomogeneousBiAgentTask.fromHomogenousBiEnvironment(supplier, stopCondition, tRange, dT);
  }
}
