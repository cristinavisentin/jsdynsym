package io.github.ericmedvet.jsdynsym.control.pong;

import io.github.ericmedvet.jsdynsym.control.HomogeneousBiAgentTask;
import io.github.ericmedvet.jsdynsym.control.Simulation;
import io.github.ericmedvet.jsdynsym.control.SimulationOutcomeDrawer;
import java.awt.*;
import java.util.SortedMap;
import java.util.function.Function;

public class SimplePongDrawer
    implements SimulationOutcomeDrawer<HomogeneousBiAgentTask.Step<double[], double[], SimplePongEnvironment.State>> {

  public SimplePongDrawer(SimplePongDrawer.Configuration configuration) {
    this.configuration = configuration;
  }

  public SimplePongDrawer() {
    this.configuration = Configuration.DEFAULT;
  }

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
      double racketsWidth,
      double marginRate) {

    public static final Configuration DEFAULT =
        new Configuration(
            Color.RED,
            Color.MAGENTA,
            Color.DARK_GRAY,
            Color.BLUE,
            0,
            0,
            0.7,
            0.95,
            1,
            0.3,
            0.6,
            0.1);
  }

  private final Configuration configuration;

  @Override
  public void drawSingle(
      Graphics2D g, double t, HomogeneousBiAgentTask.Step<double[], double[], SimplePongEnvironment.State> stateStep) {
    SimplePongEnvironment.State state = stateStep.state();
    // Compute margins and scaling factors
    double arenaWidth = state.configuration().arenaXLength();
    double arenaHeight = state.configuration().arenaYLength();
    int panelWidth = g.getClipBounds().width;
    int panelHeight = g.getClipBounds().height;
    double margin = configuration.marginRate() * Math.min(panelWidth, panelHeight);
    double scaleX = (panelWidth - 2 * margin) / arenaWidth;
    double scaleY = (panelHeight - 2 * margin) / arenaHeight;
    double scale = Math.min(scaleX, scaleY);
    double offsetX = (panelWidth - scale * arenaWidth) / 2.0;
    double offsetY = (panelHeight - scale * arenaHeight) / 2.0;
    // Helper functions for coordinate transformation
    Function<Double, Double> screenX = x -> offsetX + x * scale;
    Function<Double, Double> screenY = y -> offsetY + (arenaHeight - y) * scale;
    Function<Double, Double> screenLength = length -> length * scale;
    // Draw the arena
    g.setColor(configuration.arenaColor());
    g.setStroke(new BasicStroke((float) configuration.arenaThickness()));
    g.draw(new java.awt.geom.Rectangle2D.Double(
        screenX.apply(0.0),
        screenY.apply(arenaHeight),
        screenLength.apply(arenaWidth),
        screenLength.apply(arenaHeight)));
    // Draw the rackets
    g.setColor(configuration.racketsColor());
    for (SimplePongEnvironment.RacketState racketState :
        new SimplePongEnvironment.RacketState[] {state.lRacketState(), state.rRacketState()}) {
      double racketX = racketState.side() == SimplePongEnvironment.Side.LEFT ? -configuration.racketsWidth * 0.9 : arenaWidth;
      double racketYCenter = racketState.yCenter();
      double racketWidth = configuration.racketsWidth();
      double racketHeight = state.configuration().racketsLength();
      int alpha = (int) (configuration.racketsFillAlpha() * 255);
      g.setColor(new Color(
          configuration.racketsColor().getRed(),
          configuration.racketsColor().getGreen(),
          configuration.racketsColor().getBlue(),
          alpha));
      g.fill(new java.awt.geom.Rectangle2D.Double(
          screenX.apply(racketX),
          screenY.apply(racketYCenter + racketHeight / 2),
          screenLength.apply(racketWidth),
          screenLength.apply(racketHeight)));
    }

    // Draw the ball
    SimplePongEnvironment.BallState ballState = state.ballState();
    g.setColor(configuration.ballColor());
    int alpha = (int) (configuration.ballFillAlpha() * 255);
    g.setColor(new Color(
        configuration.ballColor().getRed(),
        configuration.ballColor().getGreen(),
        configuration.ballColor().getBlue(),
        alpha));
    double ballRadius = configuration.ballRadius();
    g.fill(new java.awt.geom.Ellipse2D.Double(
        screenX.apply(ballState.position().x() - ballRadius),
        screenY.apply(ballState.position().y() + ballRadius),
        screenLength.apply(ballRadius * 2),
        screenLength.apply(ballRadius * 2)));

    // Draw the scores
    g.setColor(configuration.infoColor());
    g.setFont(new Font("Arial", Font.BOLD, 8));
    String scoreText = String.format("L: %.0f  R: %.0f", state.lRacketState().score(), state.rRacketState().score());
    String ballVelocity = String.format(
        "BALL_V: %.0f", state.ballState().velocity().magnitude());
    String ballCollisions = String.format("#BALL_COLLISIONS: %d", state.ballState().nOfCollisions());
    g.drawString(scoreText, (float) (margin), (float) (margin * 0.9));
    g.drawString(ballVelocity, (float) (margin * 4), (float) (margin * 0.9));
    g.drawString(ballCollisions, (float) (margin * 8), (float) (margin * 0.9));
  }

  @Override
  public ImageInfo imageInfo(
      Simulation.Outcome<HomogeneousBiAgentTask.Step<double[], double[], SimplePongEnvironment.State>> stepOutcome) {
    SortedMap<Double, HomogeneousBiAgentTask.Step<double[], double[], SimplePongEnvironment.State>> snapshots =
        stepOutcome.snapshots();
    HomogeneousBiAgentTask.Step<double[], double[], SimplePongEnvironment.State> firstStep =
        snapshots.get(snapshots.firstKey());
    SimplePongEnvironment.State state = firstStep.state();
    int maxResolution = 300;
    int w;
    int h;
    if (state.configuration().arenaXLength() > state.configuration().arenaYLength()) {
      w = maxResolution;
      h = (int) (state.configuration().arenaYLength()
          / state.configuration().arenaXLength()
          * maxResolution);
      h = h + (h % 2); // it needs to be a multiple of 2
    } else {
      h = maxResolution;
      w = (int) (state.configuration().arenaXLength()
          / state.configuration().arenaYLength()
          * maxResolution);
      w = w + (w % 2); // it needs to be a multiple of 2
    }
    return new ImageInfo(w, h);
  }
}
