package io.github.ericmedvet.jsdynsym.control.pong;

import io.github.ericmedvet.jsdynsym.control.HomogeneousBiAgentTask;
import io.github.ericmedvet.jsdynsym.control.SimulationOutcomeDrawer;

import java.awt.*;

public class PongDrawer
    implements SimulationOutcomeDrawer<HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State>> {

  public record Configuration(
      Color racketsColor,
      Color ballColor,
      Color arenaColor,
      Color infoColor,
      double racketsThickness,
      double ballThickness,
      double arenaThickness,
      double racketsFillAlpha,
      double ballFillAlpha,
      double ballRadius,
      double marginRate
  ) {

    public static final Configuration DEFAULT = new Configuration(
        Color.BLUE,
        Color.MAGENTA,
        Color.DARK_GRAY,
        Color.BLUE,
        2,
        2,
        3,
        0.25,
        0.25,
        3,
        0.01
    );
  }

  private double offsetX;
  private double scale;
  private double offsetY;
  private double arenaHeight;

  // Helper per convertire coordinate da arena a schermo
  int screenX(double x) {
    return (int) (offsetX + x * scale);
  }

  int screenY(double y) {
    return (int) (offsetY + (arenaHeight - y) * scale); // Y invertito
  }

  int screenLength(double length) {
    return (int) (length * scale);
  }



  @Override
  public void drawSingle(Graphics2D g, double t, HomogeneousBiAgentTask.Step<double[], double[], PongEnvironment.State> stateStep) {

  }
}
