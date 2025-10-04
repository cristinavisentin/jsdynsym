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
import io.github.ericmedvet.jsdynsym.control.navigation.Arena;
import io.github.ericmedvet.jsdynsym.control.navigation.NavigationDrawer;
import io.github.ericmedvet.jsdynsym.control.navigation.PointNavigationDrawer;
import io.github.ericmedvet.jsdynsym.control.navigation.VectorFieldDrawer;
import io.github.ericmedvet.jsdynsym.control.pong.PongDrawer;

@Discoverable(prefixTemplate = "dynamicalSystem|dynSys|ds.drawer|d")
public class Drawers {

  private Drawers() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static NavigationDrawer navigation(
      @Param(value = "ioType", dS = "graphic") NavigationDrawer.Configuration.IOType ioType,
      @Param(value = "showSensors", dB = true) boolean showSensors
  ) {
    return new NavigationDrawer(
        new NavigationDrawer.Configuration(
            NavigationDrawer.Configuration.DEFAULT.robotColor(),
            NavigationDrawer.Configuration.DEFAULT.infoColor(),
            NavigationDrawer.Configuration.DEFAULT.sensorsColor(),
            NavigationDrawer.Configuration.DEFAULT.landmarkThickness(),
            NavigationDrawer.Configuration.DEFAULT.landmarkSize(),
            NavigationDrawer.Configuration.DEFAULT.robotThickness(),
            NavigationDrawer.Configuration.DEFAULT.robotFillAlpha(),
            NavigationDrawer.Configuration.DEFAULT.trajectoryThickness(),
            NavigationDrawer.Configuration.DEFAULT.sensorsThickness(),
            NavigationDrawer.Configuration.DEFAULT.sensorsFillAlpha(),
            ioType,
            showSensors,
            NavigationDrawer.Configuration.DEFAULT.arenaConfiguration()
        )
    );
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static PointNavigationDrawer pointNavigation(
      @Param(value = "ioType", dS = "graphic") NavigationDrawer.Configuration.IOType ioType
  ) {
    return new PointNavigationDrawer(
        new PointNavigationDrawer.Configuration(
            PointNavigationDrawer.Configuration.DEFAULT.robotColor(),
            PointNavigationDrawer.Configuration.DEFAULT.infoColor(),
            PointNavigationDrawer.Configuration.DEFAULT.robotThickness(),
            PointNavigationDrawer.Configuration.DEFAULT.robotFillAlpha(),
            PointNavigationDrawer.Configuration.DEFAULT.landmarkThickness(),
            PointNavigationDrawer.Configuration.DEFAULT.landmarkSize(),
            PointNavigationDrawer.Configuration.DEFAULT.robotDotSize(),
            PointNavigationDrawer.Configuration.DEFAULT.trajectoryThickness(),
            PointNavigationDrawer.Configuration.DEFAULT.sensorsFillAlpha(),
            ioType,
            PointNavigationDrawer.Configuration.DEFAULT.arenaConfiguration()
        )
    );
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static VectorFieldDrawer vectorField(
      @Param(value = "arena", dNPM = "empty") Arena.Prepared arena
  ) {
    return new VectorFieldDrawer(arena.arena(), VectorFieldDrawer.Configuration.DEFAULT);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static PongDrawer pong() {
    return new PongDrawer(PongDrawer.Configuration.DEFAULT);
  }
}
