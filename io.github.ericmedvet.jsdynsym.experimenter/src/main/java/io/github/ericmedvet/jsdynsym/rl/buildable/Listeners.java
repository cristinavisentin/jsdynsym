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

import io.github.ericmedvet.jnb.core.Alias;
import io.github.ericmedvet.jnb.core.Discoverable;

@Discoverable(prefixTemplate = "rl.listener|l")
@Alias(
    name = "console", value = // spotless:off
    """       
       listener.console(
         defaultEFunctions = [
           rl.f.nOfEpisodes();
           rl.f.nOfSteps();
           rl.f.elapsedSecs();
           ds.f.cumulatedReward(of = rl.f.lastOutcome())
         ];
         defaultKFunctions = [
           f.interpolated(s = "{agent.name}")
         ]
       )
       """ // spotless:on
)
public class Listeners {

  private Listeners() {
  }

}
