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
import io.github.ericmedvet.jsdynsym.control.navigation.NavigationArena;
import io.github.ericmedvet.jsdynsym.control.navigation.NavigationEnvironment;
import io.github.ericmedvet.jsdynsym.control.navigation.VariableSensorPositionsNavigation;
import java.util.List;
import java.util.random.RandomGenerator;

@Discoverable(prefixTemplate = "dynamicalSystem|dynSys|ds.simulation|sim|s")
public class Simulations {
  private Simulations() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static VariableSensorPositionsNavigation variableSensorPositionsNavigation(
      @Param(value = "name", iS = "vs[{nOfSensors}]-nav-{arena}") String name,
      @Param(value = "initialRobotDirectionRange", dNPM = "m.range(min=0;max=0)") DoubleRange initialRobotDirectionRange,
      @Param(value = "robotRadius", dD = 0.05) double robotRadius,
      @Param(value = "robotMaxV", dD = 0.01) double robotMaxV,
      @Param(value = "nOfSensors", dI = 5) int nOfSensors,
      @Param(value = "sensorRange", dD = .5) double sensorRange,
      @Param(value = "senseTarget", dB = true) boolean senseTarget,
      @Param(value = "arena", dNPM = "ds.arena.prepared()") NavigationArena arena,
      @Param(value = "rescaleInput", dB = true) boolean rescaleInput,
      @Param("relativeV") boolean relativeSpeed,
      @Param(value = "sortAngles", dB = true) boolean sortAngles,
      @Param(value = "randomGenerator", dNPM = "m.defaultRG()") RandomGenerator randomGenerator
  ) {
    return new VariableSensorPositionsNavigation(
        new NavigationEnvironment.Configuration(
            initialRobotDirectionRange,
            robotRadius,
            robotMaxV,
            List.of(),
            sensorRange,
            senseTarget,
            arena,
            rescaleInput,
            relativeSpeed,
            randomGenerator
        ),
        nOfSensors,
        sortAngles
    );
  }
}
