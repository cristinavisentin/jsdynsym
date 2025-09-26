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
package io.github.ericmedvet.jsdynsym.control.navigation;

import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.Grid;
import io.github.ericmedvet.jnb.datastructure.Grid.Entry;
import io.github.ericmedvet.jnb.datastructure.Grid.Key;
import io.github.ericmedvet.jsdynsym.control.geometry.Segment;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

public interface NavigationArena extends Arena {

  Character OBSTACLE_CHAR = 'w';
  Character EMPTY_CHAR = ' ';
  Character STARTING_POINT_CHAR = 's';
  Character TARGET_POINT_CHAR = 't';

  DoubleRange initialRobotXRange();

  DoubleRange initialRobotYRange();

  DoubleRange targetXRange();

  DoubleRange targetYRange();

  static NavigationArena of(
      Arena arena,
      DoubleRange initialRobotXRange,
      DoubleRange initialRobotYRange,
      DoubleRange targetXRange,
      DoubleRange targetYRange
  ) {
    record HardNavigationArena(
        double xExtent, double yExtent, List<Segment> obstacles,
        DoubleRange initialRobotXRange,
        DoubleRange initialRobotYRange,
        DoubleRange targetXRange,
        DoubleRange targetYRange
    ) implements NavigationArena {

    }
    return new HardNavigationArena(
        arena.xExtent(),
        arena.yExtent(),
        arena.obstacles(),
        initialRobotXRange,
        initialRobotYRange,
        targetXRange,
        targetYRange
    );
  }

  static NavigationArena fromGrid(Grid<Character> grid, double sideLength, boolean diagonal) {
    if (!Set.of(EMPTY_CHAR, OBSTACLE_CHAR, STARTING_POINT_CHAR, TARGET_POINT_CHAR)
        .containsAll(grid.values())) {
      throw new IllegalArgumentException(
          "Unknown chars in the grid: %s not in known set %s".formatted(
              new TreeSet<>(grid.values()),
              Set.of(EMPTY_CHAR, OBSTACLE_CHAR, STARTING_POINT_CHAR, TARGET_POINT_CHAR)
          )
      );
    }
    if (grid.values().stream().filter(STARTING_POINT_CHAR::equals).count() != 1) {
      throw new IllegalArgumentException(
          "Grid with wrong number of starting point char %s: %d instead of 1".formatted(
              STARTING_POINT_CHAR,
              grid.values().stream().filter(STARTING_POINT_CHAR::equals).count()
          )
      );
    }
    if (grid.values().stream().filter(TARGET_POINT_CHAR::equals).count() != 1) {
      throw new IllegalArgumentException(
          "Grid with wrong number of target point char %s: %d instead of 1".formatted(
              TARGET_POINT_CHAR,
              grid.values().stream().filter(TARGET_POINT_CHAR::equals).count()
          )
      );
    }
    Key startingPoint = grid.entries()
        .stream()
        .filter(e -> e.value().equals(STARTING_POINT_CHAR))
        .map(Entry::key)
        .findFirst()
        .orElse(new Key(0, 0));
    Key targetPoint = grid.entries()
        .stream()
        .filter(e -> e.value().equals(TARGET_POINT_CHAR))
        .map(Entry::key)
        .findFirst()
        .orElse(new Key(0, 0));
    return of(
        Arena.fromGrid(grid.map(c -> OBSTACLE_CHAR == c), sideLength, diagonal),
        new DoubleRange(startingPoint.x() + sideLength / 2d, startingPoint.x() + sideLength / 2d),
        new DoubleRange(startingPoint.y() + sideLength / 2d, startingPoint.y() + sideLength / 2d),
        new DoubleRange(targetPoint.x() + sideLength / 2d, targetPoint.x() + sideLength / 2d),
        new DoubleRange(targetPoint.y() + sideLength / 2d, targetPoint.y() + sideLength / 2d)
    );

  }
}
