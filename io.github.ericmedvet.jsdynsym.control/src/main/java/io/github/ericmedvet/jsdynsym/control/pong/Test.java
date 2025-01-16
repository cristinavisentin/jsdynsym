package io.github.ericmedvet.jsdynsym.control.pong;

import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.Pair;
import io.github.ericmedvet.jsdynsym.control.HomogeneousBiAgentTask;
import io.github.ericmedvet.jsdynsym.control.Simulation;
import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;
import io.github.ericmedvet.jviz.core.drawer.VideoBuilder;
import io.github.ericmedvet.jviz.core.util.VideoUtils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class Test {
  public static void main(String[] args){
    singleTest(200, 0.05, new Pair<>(10.0, 10.0), "../pong.mp4");

    //multipleTestsWithConfusionMatrix(2000, 0.05, 101, Arrays.asList(0.0, 0.5, 10.0), "../pong_confusion_matrix.csv");
  }

  private static void multipleTestsWithConfusionMatrix(double finalTime, double dT, double finalScore, List<Double> agentsSpeedsDumpingConstants, String csvFilePath){
    HomogeneousBiAgentTask<DynamicalSystem<double[], double[], ?>, double[], double[], PongEnvironment.State>
        task = HomogeneousBiAgentTask.fromHomogenousBiEnvironment(
        new PongEnvironment(PongEnvironment.Configuration.DEFAULT),
        s -> s.lRacketScore() + s.rRacketScore() > finalScore,
        new DoubleRange(0, finalTime),
        dT
    );
    try (FileWriter writer = new FileWriter(csvFilePath)) {
      writer.write("Agent vs Agent;");
      for (Double agentsSpeed : agentsSpeedsDumpingConstants) {
        writer.write("Agent " + agentsSpeed + ";");
      }
      writer.write("\n");
      for (Double agentsSpeed_i : agentsSpeedsDumpingConstants) {
        writer.write("Agent " + agentsSpeed_i + ";");
        for (Double agentsSpeed_j : agentsSpeedsDumpingConstants) {
          Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>
              outcome = task.simulate(new SimplePongAgent(agentsSpeed_i), new SimplePongAgent(agentsSpeed_j));
          String score = getFinalScore(outcome);
          writer.write(score + ";");
        }
        writer.write("\n");
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static void singleTest(double finalTime, double dT, Pair<Double, Double> agentsSpeedsDumpingConstants, String pathName) {
    HomogeneousBiAgentTask<DynamicalSystem<double[], double[], ?>, double[], double[], PongEnvironment.State>
        dynamicalSystemStateHomogeneousBiAgentTask = HomogeneousBiAgentTask.fromHomogenousBiEnvironment(
        new PongEnvironment(PongEnvironment.Configuration.DEFAULT),
        x -> false,
        new DoubleRange(0, finalTime),
        dT
    );
    Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>>
        outcome = dynamicalSystemStateHomogeneousBiAgentTask.simulate(new SimplePongAgent(agentsSpeedsDumpingConstants.first()), new SimplePongAgent(agentsSpeedsDumpingConstants.second()));
    System.out.println("L_Score: " + outcome.snapshots().get(outcome.snapshots().lastKey()).state().lRacketScore());
    System.out.println("R_Score: " + outcome.snapshots().get(outcome.snapshots().lastKey()).state().rRacketScore());
    PongDrawer pongDrawer = new PongDrawer();
    // pongDrawer.show(outcome);
    pongDrawer.videoBuilder().save(new VideoBuilder.VideoInfo(400, 200, VideoUtils.EncoderFacility.DEFAULT), new File(pathName), outcome);
  }

  private static String getFinalScore(Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>> outcome) {
    var snapshots = outcome.snapshots();
    var finalStep = snapshots.get(snapshots.lastKey());
    double lScore = finalStep.state().lRacketScore();
    double rScore = finalStep.state().rRacketScore();
    return lScore + " - " + rScore;
  }
}
