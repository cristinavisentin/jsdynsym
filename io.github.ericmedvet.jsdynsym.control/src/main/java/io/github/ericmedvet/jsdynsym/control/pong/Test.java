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
    HomogeneousBiAgentTask<DynamicalSystem<double[], double[], ?>, double[], double[], SimplePongEnvironment.State>
        dynamicalSystemStateHomogeneousBiAgentTask = HomogeneousBiAgentTask.fromHomogenousBiEnvironment(
        () -> new SimplePongEnvironment(SimplePongEnvironment.Configuration.DEFAULT),
        s -> s.lRacketState().score() + s.rRacketState().score() >= 101,
        new DoubleRange(0, 20),
        0.05);
    Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], SimplePongEnvironment.State>> outcome =
        dynamicalSystemStateHomogeneousBiAgentTask.simulate(new Pair<> (new SimplePongAgent(1), new SimplePongAgent(1)));
    System.out.println("L_Score: "
        + outcome.snapshots().get(outcome.snapshots().lastKey()).state().lRacketState().score());
    System.out.println("R_Score: "
        + outcome.snapshots().get(outcome.snapshots().lastKey()).state().rRacketState().score());
    SimplePongDrawer pongDrawer = new SimplePongDrawer();
    // pongDrawer.show(outcome);
    pongDrawer.videoBuilder().save(new File("../simple-pong.mp4"), outcome);

    // singleTest(200, 0.02, 6, new Pair<>(new PongAgent(1), new PongAgent(1)), "../pong_1.mp4");

    // multipleTestsWithConfusionMatrix(2000, 0.05, 101, Arrays.asList(0.0, 0.5, 10.0),
    // "../pong_confusion_matrix.csv");

    // List<DynamicalSystem<double[], double[], ?>> agentsList =
    //    List.of(new SimplePongAgent(0), new SimplePongAgent(0.1), new MaxSpeedPongAgent());
    // multipleTestsWithConfusionMatrixAndTimes(2000, 0.05, 11, agentsList, 100, "../pong_scores-CM.csv",
    // "../pong_times-CM.csv");
  }

  private static void multipleTestsWithConfusionMatrixAndTimes(
      double finalTime,
      double dT,
      double finalScore,
      List<DynamicalSystem<double[], double[], ?>> agentsList,
      int repetitions,
      String scoresCsvFilePath,
      String timesCsvFilePath) {
    HomogeneousBiAgentTask<DynamicalSystem<double[], double[], ?>, double[], double[], PongEnvironment.State> task =
        HomogeneousBiAgentTask.fromHomogenousBiEnvironment(
            () -> new PongEnvironment(PongEnvironment.Configuration.DEFAULT),
            s -> s.lRacketScore() + s.rRacketScore() >= finalScore,
            new DoubleRange(0, finalTime),
            dT);
    try (FileWriter scoresWriter = new FileWriter(scoresCsvFilePath);
        FileWriter timesWriter = new FileWriter(timesCsvFilePath)) {
      // Intestazioni CSV
      scoresWriter.write("Agent vs Agent;");
      timesWriter.write("Agent vs Agent;");
      for (DynamicalSystem<double[], double[], ?> agent : agentsList) {
        scoresWriter.write(agent.toString() + ";");
        timesWriter.write(agent.toString() + ";");
      }
      scoresWriter.write("\n");
      timesWriter.write("\n");

      // Simulazioni
      for (DynamicalSystem<double[], double[], ?> agent_i : agentsList) {
        scoresWriter.write(agent_i.toString() + ";");
        timesWriter.write(agent_i.toString() + ";");
        for (DynamicalSystem<double[], double[], ?> agent_j : agentsList) {
          double totalLScore = 0.0;
          double totalRScore = 0.0;
          double totalTime = 0.0;

          for (int r = 0; r < repetitions; r++) {
            long startTime = System.nanoTime();
            Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>
                outcome = task.simulate(agent_i, agent_j);
            long endTime = System.nanoTime();

            var snapshots = outcome.snapshots();
            var finalStep = snapshots.get(snapshots.lastKey());
            double lScore = finalStep.state().lRacketScore();
            double rScore = finalStep.state().rRacketScore();

            totalLScore += lScore;
            totalRScore += rScore;
            totalTime += (endTime - startTime) / 1e9;
          }

          // Calcolo delle medie
          double averageLScore = totalLScore / repetitions;
          double averageRScore = totalRScore / repetitions;
          double averageTime = totalTime / repetitions;

          // Scrittura nei file
          scoresWriter.write(averageLScore + " - " + averageRScore + ";");
          timesWriter.write(averageTime + ";");
        }
        scoresWriter.write("\n");
        timesWriter.write("\n");
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static void multipleTestsWithConfusionMatrix(
      double finalTime,
      double dT,
      double finalScore,
      List<DynamicalSystem<double[], double[], ?>> agentsSpeedsDumpingConstants,
      String csvFilePath) {
    HomogeneousBiAgentTask<DynamicalSystem<double[], double[], ?>, double[], double[], PongEnvironment.State> task =
        HomogeneousBiAgentTask.fromHomogenousBiEnvironment(
            () -> new PongEnvironment(PongEnvironment.Configuration.DEFAULT),
            s -> s.lRacketScore() + s.rRacketScore() >= finalScore,
            new DoubleRange(0, finalTime),
            dT);
    try (FileWriter writer = new FileWriter(csvFilePath)) {
      writer.write("Agent vs Agent;");
      for (DynamicalSystem<double[], double[], ?> agentsSpeed : agentsSpeedsDumpingConstants) {
        writer.write("Agent " + agentsSpeed.toString() + ";");
      }
      writer.write("\n");
      for (DynamicalSystem<double[], double[], ?> agentsSpeed_i : agentsSpeedsDumpingConstants) {
        writer.write("Agent " + agentsSpeed_i.toString() + ";");
        for (DynamicalSystem<double[], double[], ?> agentsSpeed_j : agentsSpeedsDumpingConstants) {
          Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>> outcome =
              task.simulate(agentsSpeed_i, agentsSpeed_j);
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
            () -> new PongEnvironment(PongEnvironment.Configuration.DEFAULT),
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
