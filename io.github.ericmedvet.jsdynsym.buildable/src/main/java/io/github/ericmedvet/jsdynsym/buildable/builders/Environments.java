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

import io.github.ericmedvet.jnb.core.Discoverable;
import io.github.ericmedvet.jnb.core.Param;
import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jsdynsym.control.navigation.Arena;
import io.github.ericmedvet.jsdynsym.control.navigation.NavigationEnvironment;
import io.github.ericmedvet.jsdynsym.control.navigation.PointNavigationEnvironment;
import io.github.ericmedvet.jsdynsym.control.pong.PongEnvironment;
import java.util.random.RandomGenerator;

@Discoverable(prefixTemplate = "dynamicalSystem|dynSys|ds.environment|env|e")
public class Environments {
  private Environments() {
  }

  public static PongEnvironment pong(
      @Param(value = "name", iS = "pong") String name,
      @Param(value = "racketsInitialYRange", dNPM = "m.range(min=22.0;max=28.0)") DoubleRange racketsInitialYRange,
      @Param(value = "racketsLength", dD = 5.0) double racketsLength,
      @Param(value = "racketsMaxDeltaPosition", dD = 0.5) double racketsMaxDeltaPosition,
      @Param(value = "ballInitialVelocity", dD = 20.0) double ballInitialVelocity,
      @Param(value = "ballMaxVelocity", dD = 50.0) double ballMaxVelocity,
      @Param(value = "ballInitialAngleRange", dNPM = "m.range(min=-0.4;max=0.4)") DoubleRange ballInitialAngleRange,
      @Param(value = "ballAccelerationRate", dD = 1.1) double ballAccelerationRate,
      @Param(value = "maxPercentageAngleAdjustment", dD = 0.1) double maxPercentageAngleAdjustment,
      @Param(value = "arenaXLength", dD = 60.0) double arenaXLength,
      @Param(value = "arenaYLength", dD = 50.0) double arenaYLength,
      @Param(value = "precision", dD = 1e-5) double precision,
      @Param(value = "randomGenerator", dNPM = "m.defaultRG()") RandomGenerator randomGenerator
  ) {
    return new PongEnvironment(
        new PongEnvironment.Configuration(
            racketsInitialYRange, //
            racketsLength, //
            racketsMaxDeltaPosition, //
            ballInitialVelocity,
            ballMaxVelocity,
            ballInitialAngleRange,
            ballAccelerationRate,
            maxPercentageAngleAdjustment,
            arenaXLength,
            arenaYLength,
            precision,
            randomGenerator
        )
    );
  }

  @SuppressWarnings("unused")
  public static NavigationEnvironment navigation(
      @Param(value = "name", iS = "nav-{arena}") String name,
      @Param(value = "initialRobotXRange", dNPM = "m.range(min=0.45;max=0.55)") DoubleRange initialRobotXRange,
      @Param(value = "initialRobotYRange", dNPM = "m.range(min=0.8;max=0.85)") DoubleRange initialRobotYRange,
      @Param(value = "initialRobotDirectionRange", dNPM = "m.range(min=0;max=0)") DoubleRange initialRobotDirectionRange,
      @Param(value = "targetXRange", dNPM = "m.range(min=0.5;max=0.5)") DoubleRange targetXRange,
      @Param(value = "targetYRange", dNPM = "m.range(min=0.15;max=0.15)") DoubleRange targetYRange,
      @Param(value = "robotRadius", dD = 0.05) double robotRadius,
      @Param(value = "robotMaxV", dD = 0.01) double robotMaxV,
      @Param(value = "sensorsAngleRange", dNPM = "m.range(min=-1.57;max=1.57)") DoubleRange sensorsAngleRange,
      @Param(value = "nOfSensors", dI = 5) int nOfSensors,
      @Param(value = "sensorRange", dD = .5) double sensorRange,
      @Param(value = "senseTarget", dB = true) boolean senseTarget,
      @Param(value = "arena", dS = "empty") Arena.Prepared arena,
      @Param(value = "rescaleInput", dB = true) boolean rescaleInput,
      @Param(value = "randomGenerator", dNPM = "m.defaultRG()") RandomGenerator randomGenerator
  ) {
    return new NavigationEnvironment(
        new NavigationEnvironment.Configuration(
            initialRobotXRange,
            initialRobotYRange,
            initialRobotDirectionRange,
            targetXRange,
            targetYRange,
            robotRadius,
            robotMaxV,
            sensorsAngleRange.points(nOfSensors).boxed().toList(),
            sensorRange,
            senseTarget,
            arena.arena(),
            rescaleInput,
            randomGenerator
        )
    );
  }

  @SuppressWarnings("unused")
  public static PointNavigationEnvironment pointNavigation(
      @Param(value = "name", iS = "nav-{arena}") String name,
      @Param(value = "initialRobotXRange", dNPM = "m.range(min=0.45;max=0.55)") DoubleRange initialRobotXRange,
      @Param(value = "initialRobotYRange", dNPM = "m.range(min=0.8;max=0.85)") DoubleRange initialRobotYRange,
      @Param(value = "targetXRange", dNPM = "m.range(min=0.5;max=0.5)") DoubleRange targetXRange,
      @Param(value = "targetYRange", dNPM = "m.range(min=0.15;max=0.15)") DoubleRange targetYRange,
      @Param(value = "robotMaxV", dD = 0.01) double robotMaxV,
      @Param(value = "collisionBlock", dD = 0.005) double collisionBlock,
      @Param(value = "arena", dS = "empty") Arena.Prepared arena,
      @Param(value = "rescaleInput", dB = true) boolean rescaleInput,
      @Param(value = "randomGenerator", dNPM = "m.defaultRG()") RandomGenerator randomGenerator
  ) {
    return new PointNavigationEnvironment(
        new PointNavigationEnvironment.Configuration(
            initialRobotXRange,
            initialRobotYRange,
            targetXRange,
            targetYRange,
            robotMaxV,
            collisionBlock,
            arena.arena(),
            rescaleInput,
            randomGenerator
        )
    );
  }
}
