/*-
 * ========================LICENSE_START=================================
 * jsdynsym-experimenter
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
package io.github.ericmedvet.jsdynsym.rl;

import io.github.ericmedvet.jnb.core.Interpolator;
import io.github.ericmedvet.jnb.core.MapNamedParamMap;
import io.github.ericmedvet.jnb.core.ParamMap;
import java.util.Map;

public class Utils {

  private Utils() {
  }

  public static String interpolate(String format, Experiment experiment, Run<?, ?, ?, ?, ?, ?> run) {
    ParamMap map = new MapNamedParamMap("experiment", Map.of());
    if (experiment != null) {
      map = experiment.map();
    }
    if (run != null) {
      map = map.with(
          "run",
          run.map().with("nOfEpisodes", run.index())
      );
    }
    return Interpolator.interpolate(format, map, "_");
  }

}
