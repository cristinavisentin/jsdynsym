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
package io.github.ericmedvet.jsdynsym.control.pong;

import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.Pair;
import io.github.ericmedvet.jsdynsym.control.HomogeneousBiAgentTask;
import io.github.ericmedvet.jsdynsym.control.Simulation;
import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class Test {
  public static void main(String[] args) {
    singleTest(500, 0.05, 11, new Pair<>(new SimplePongAgent(0.1), new MaxSpeedPongAgent()), "../pong.mp4");

    // multipleTestsWithConfusionMatrix(2000, 0.05, 101, Arrays.asList(0.0, 0.5, 10.0),
    // "../pong_confusion_matrix.csv");
  }

  private static void multipleTestsWithConfusionMatrix(
      double finalTime,
      double dT,
      double finalScore,
      List<Double> agentsSpeedsDumpingConstants,
      String csvFilePath) {
    HomogeneousBiAgentTask<DynamicalSystem<double[], double[], ?>, double[], double[], PongEnvironment.State> task =
        HomogeneousBiAgentTask.fromHomogenousBiEnvironment(
            new PongEnvironment(PongEnvironment.Configuration.DEFAULT),
            s -> s.lRacketScore() + s.rRacketScore() >= finalScore,
            new DoubleRange(0, finalTime),
            dT);
    try (FileWriter writer = new FileWriter(csvFilePath)) {
      writer.write("Agent vs Agent;");
      for (Double agentsSpeed : agentsSpeedsDumpingConstants) {
        writer.write("Agent " + agentsSpeed + ";");
      }
      writer.write("\n");
      for (Double agentsSpeed_i : agentsSpeedsDumpingConstants) {
        writer.write("Agent " + agentsSpeed_i + ";");
        for (Double agentsSpeed_j : agentsSpeedsDumpingConstants) {
          Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>> outcome =
              task.simulate(new SimplePongAgent(agentsSpeed_i), new SimplePongAgent(agentsSpeed_j));
          String score = getFinalScore(outcome);
          writer.write(score + ";");
        }
        writer.write("\n");
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static void singleTest(
      double finalTime,
      double dT,
      double finalScore,
      Pair<DynamicalSystem<double[], double[], ?>, DynamicalSystem<double[], double[], ?>>
          agentsSpeedsDumpingConstants,
      String pathName) {
    HomogeneousBiAgentTask<DynamicalSystem<double[], double[], ?>, double[], double[], PongEnvironment.State>
        dynamicalSystemStateHomogeneousBiAgentTask = HomogeneousBiAgentTask.fromHomogenousBiEnvironment(
            new PongEnvironment(PongEnvironment.Configuration.DEFAULT),
            s -> s.lRacketScore() + s.rRacketScore() >= finalScore,
            new DoubleRange(0, finalTime),
            dT);
    Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>> outcome =
        dynamicalSystemStateHomogeneousBiAgentTask.simulate(agentsSpeedsDumpingConstants);
    System.out.println("L_Score: "
        + outcome.snapshots().get(outcome.snapshots().lastKey()).state().lRacketScore());
    System.out.println("R_Score: "
        + outcome.snapshots().get(outcome.snapshots().lastKey()).state().rRacketScore());
    PongDrawer pongDrawer = new PongDrawer();
    // pongDrawer.show(outcome);
    pongDrawer.videoBuilder().save(new File(pathName), outcome);
  }

  private static String getFinalScore(
      Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>> outcome) {
    var snapshots = outcome.snapshots();
    var finalStep = snapshots.get(snapshots.lastKey());
    double lScore = finalStep.state().lRacketScore();
    double rScore = finalStep.state().rRacketScore();
    return lScore + " - " + rScore;
  }
}
