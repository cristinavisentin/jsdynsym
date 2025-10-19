/*-
 * ========================LICENSE_START=================================
 * jsdynsym-core
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

package io.github.ericmedvet.jsdynsym.core.rl;

import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;

public interface ReinforcementLearningAgent<I, O, S> extends DynamicalSystem<ReinforcementLearningAgent.RewardedInput<I>, O, S> {
  record RewardedInput<I>(I input, double reward) {}

  O step(double t, I input, double reward);

  @Override
  default O step(double t, RewardedInput<I> rewardedInput) {
    return step(t, rewardedInput.input(), rewardedInput.reward());
  }

  static <I, O, S> ReinforcementLearningAgent<I, O, S> from(DynamicalSystem<I, O, S> dynamicalSystem) {
    return new ReinforcementLearningAgent<>() {
      @Override
      public O step(double t, I input, double reward) {
        return dynamicalSystem.step(t, input);
      }

      @Override
      public S getState() {
        return dynamicalSystem.getState();
      }

      @Override
      public void reset() {
        dynamicalSystem.reset();
      }

      @Override
      public String toString() {
        return dynamicalSystem.toString();
      }
    };
  }
}
