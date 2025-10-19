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
package io.github.ericmedvet.jsdynsym.rl.buildable;

import io.github.ericmedvet.jgea.core.util.Naming;
import io.github.ericmedvet.jgea.experimenter.Utils;
import io.github.ericmedvet.jnb.core.Cacheable;
import io.github.ericmedvet.jnb.core.Discoverable;
import io.github.ericmedvet.jnb.core.Param;
import io.github.ericmedvet.jnb.datastructure.NamedFunction;
import io.github.ericmedvet.jnb.datastructure.TriConsumer;
import io.github.ericmedvet.jsdynsym.rl.Experiment;
import io.github.ericmedvet.jsdynsym.rl.Run;
import java.util.function.Function;

@Discoverable(prefixTemplate = "rl.consumer|c")
public class Consumers {

  private Consumers() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static <X, O> TriConsumer<X, Run<?, ?, ?, ?, ?, ?>, Experiment> saver(
      @Param(value = "of", dNPM = "f.identity()") Function<X, O> f,
      @Param(value = "overwrite") boolean overwrite,
      @Param(value = "path", dS = "run-{run.index:%04d}") String filePathTemplate,
      @Param(value = "suffix", dS = "") String suffix
  ) {
    return Naming.named(
        "saver[%s;%s]".formatted(
            NamedFunction.name(f),
            filePathTemplate + (overwrite ? "(*)" : "")
        ),
        (TriConsumer<X, Run<?, ?, ?, ?, ?, ?>, Experiment>) (x, run, experiment) -> Utils.save(
            f.apply(x),
            io.github.ericmedvet.jsdynsym.rl.Utils.interpolate(filePathTemplate, experiment, run) + suffix,
            overwrite
        )
    );
  }

}
