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
import io.github.ericmedvet.jnb.datastructure.Grid;
import io.github.ericmedvet.jsdynsym.control.navigation.Arena;
import io.github.ericmedvet.jsdynsym.control.navigation.NavigationArena;
import java.util.regex.Pattern;

@Discoverable(prefixTemplate = "dynamicalSystem|dynSys|ds.arena")
public class Arenas {

  private Arenas() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static NavigationArena prepared(
      @Param(value = "name", dS = "empty") Arena.Prepared prepared,
      @Param(value = "initialRobotXRange", dNPM = "m.range(min=0.45;max=0.55)") DoubleRange initialRobotXRange,
      @Param(value = "initialRobotYRange", dNPM = "m.range(min=0.8;max=0.85)") DoubleRange initialRobotYRange,
      @Param(value = "targetXRange", dNPM = "m.range(min=0.5;max=0.5)") DoubleRange targetXRange,
      @Param(value = "targetYRange", dNPM = "m.range(min=0.15;max=0.15)") DoubleRange targetYRange
  ) {
    return NavigationArena.of(prepared.arena(), initialRobotXRange, initialRobotYRange, targetXRange, targetYRange);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static NavigationArena fromString(
      @Param(value = "name", dS = "custom") String name,
      @Param("s") String s,
      @Param(value = "l", dD = 0.1) double sideLength,
      @Param("diagonal") boolean diagonal,
      @Param(value = "emptyChar", dS = " ") String emptyString,
      @Param(value = "obstacleChar", dS = "w") String obstacleString,
      @Param(value = "startChar", dS = "s") String startString,
      @Param(value = "targetChar", dS = "t") String targetString,
      @Param(value = "separatorChar", dS = "|") String separatorString
  ) {
    s = s
        .replace(emptyString.charAt(0), NavigationArena.EMPTY_CHAR)
        .replace(obstacleString.charAt(0), NavigationArena.OBSTACLE_CHAR)
        .replace(startString.charAt(0), NavigationArena.STARTING_POINT_CHAR)
        .replace(targetString.charAt(0), NavigationArena.TARGET_POINT_CHAR);
    String[] lines = s.split(Pattern.quote(separatorString.substring(0, 1)));
    Grid<Character> grid = Grid.create(
        lines[0].length(),
        lines.length,
        (x, y) -> lines[y].charAt(x)
    );
    return NavigationArena.fromGrid(
        grid,
        sideLength,
        diagonal
    );
  }
}
