package io.github.ericmedvet.jsdynsym.control.pong;

import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import io.github.ericmedvet.jnb.datastructure.Pair;
import io.github.ericmedvet.jsdynsym.control.HomogeneousBiEnvironmentWithExample;
import io.github.ericmedvet.jsdynsym.control.geometry.Point;
import io.github.ericmedvet.jsdynsym.control.geometry.Rectangle;
import io.github.ericmedvet.jsdynsym.control.geometry.Segment;
import io.github.ericmedvet.jsdynsym.core.DynamicalSystem;
import io.github.ericmedvet.jsdynsym.core.numerical.NumericalStatelessSystem;

import java.util.List;
import java.util.Random;
import java.util.random.RandomGenerator;

public class SimplePongEnvironment implements HomogeneousBiEnvironmentWithExample<double[], double[], SimplePongEnvironment.State> {

  private final Configuration configuration;
  private State state;
  private double previousTime;
  private Rectangle arena;

  public SimplePongEnvironment(Configuration configuration) {
    this.configuration = configuration;
    this.arena = new Rectangle(new Point(0.0, configuration.arenaYLength), new Point(configuration.arenaXLength, 0));
    reset();
  }

  public record Configuration(
      DoubleRange racketsInitialYRange,
      double racketsLength,
      double racketsMaxDeltaY,
      double ballInitialVelocity,
      double ballMaxVelocity,
      DoubleRange ballInitialAngleRange,
      double ballAccelerationRatePerPoint, // The ball accelerates during a point
      double arenaXLength,
      double arenaYLength,
      RandomGenerator randomGenerator) {
    public static final SimplePongEnvironment.Configuration DEFAULT = new SimplePongEnvironment.Configuration(
        new DoubleRange(22, 28),
        5,
        1.5,
        30,
        70,
        new DoubleRange(-Math.PI / 8, Math.PI / 8),
        1.01,
        60,
        50,
        new Random());
  }

  public record State(
      Configuration configuration,
      RacketState lRacketState,
      RacketState rRacketState,
      BallState ballState
  ) {
  }

  public enum Side {
    LEFT,
    RIGHT
  }

  private enum ArenaObject {
    L_RACKET,
    R_RACKET,
    HORIZONTAL_EDGES,
    NONE
  }

  public record RacketState(double yCenter, int nOfBallCollisions, int score, Side side) {
    RacketState updatePosition(double yCenter) {
      return new RacketState(yCenter, this.nOfBallCollisions, this.score, this.side);
    }

    RacketState incrementNofBallCollisions() {
      return new RacketState(this.yCenter, this.nOfBallCollisions + 1, this.score, this.side);
    }

    RacketState incrementScore() {
      return new RacketState(this.yCenter, this.nOfBallCollisions, this.score + 1, this.side);
    }
  }

  public record BallState(Point position, Point velocity) {
  }

  private double[] getNormalizedObservation(RacketState racketState, BallState ballState) {
    double normalizedBallXVelocity = ballState.velocity().x() / configuration.ballMaxVelocity;
    double normalizedBallYVelocity = ballState.velocity().y() / configuration.ballMaxVelocity;
    return new double[]{
        racketState.yCenter / configuration.arenaYLength,
        ballState.position().x() / configuration.arenaXLength,
        ballState.position().y() / configuration.arenaYLength,
        normalizedBallXVelocity,
        normalizedBallYVelocity
    };
  }

  public int nOfInputsPerAgent() {
    return 1;
  }

  public int nOfObservationsPerAgent() {
    return 5;
  }

  @Override
  public DynamicalSystem<double[], double[], ?> example() {
    return NumericalStatelessSystem.from(nOfObservationsPerAgent(), nOfInputsPerAgent(), (t, in) -> new double[nOfInputsPerAgent()]);
  }

  @Override
  public Pair<double[], double[]> defaultActions() {
    return new Pair<>(new double[nOfInputsPerAgent()], new double[nOfInputsPerAgent()]);
  }

  @Override
  public State getState() {
    return state;
  }

  @Override
  public void reset() {
    state = new State(
        configuration,
        new RacketState(
            configuration.racketsInitialYRange.denormalize(configuration.randomGenerator.nextDouble()),
            0,
            0,
            Side.LEFT),
        new RacketState(
            configuration.racketsInitialYRange.denormalize(configuration.randomGenerator.nextDouble()),
            0,
            0,
            Side.RIGHT),
        new BallState(
            new Point(configuration.arenaXLength / 2.0, configuration.arenaYLength / 2.0),
            randomBallVelocity()));
    previousTime = 0.0;
  }

  @Override
  public Pair<double[], double[]> step(double t, Pair<double[], double[]> normalizedActions) {
    if (normalizedActions.first().length != nOfInputsPerAgent()) {
      throw new IllegalArgumentException("Left agent action has wrong number of elements: %d found, %d expected"
          .formatted(normalizedActions.first().length, nOfInputsPerAgent()));
    }
    if (normalizedActions.second().length != nOfInputsPerAgent()) {
      throw new IllegalArgumentException("Right agent action has wrong number of elements: %d found, %d expected"
          .formatted(normalizedActions.second().length, nOfInputsPerAgent()));
    }
    // clip and denormalize inputs
    double lAction = DoubleRange.SYMMETRIC_UNIT.clip(normalizedActions.first()[0]) * configuration.racketsMaxDeltaY;
    double rAction = DoubleRange.SYMMETRIC_UNIT.clip(normalizedActions.second()[0]) * configuration.racketsMaxDeltaY;
    // update time
    double deltaTime = t - previousTime;
    previousTime = t;
    // update rackets positions
    RacketState updatedLRacketState = updateRacketPosition(state.lRacketState, lAction);
    RacketState updatedRRacketState = updateRacketPosition(state.rRacketState, rAction);

    BallState updatedBallState = updateBallState(state.ballState, deltaTime);
    Segment ballTrajectory = getAsSegment(state.ballState, updatedBallState);
    Point lRacketCollision = ballTrajectory.intersection(getAsSegment(updatedLRacketState));
    Point rRacketCollision = ballTrajectory.intersection(getAsSegment(updatedRRacketState));
    List<Point> arenaHorizontalEdgesCollisions = arena.horizontalEdgesIntersections(ballTrajectory);
    Point arenaHorizontalEdgesCollision = arenaHorizontalEdgesCollisions.isEmpty() ? null : arenaHorizontalEdgesCollisions.getFirst();

    Point previousBallPosition = state.ballState.position;
    ArenaObject closestCollidingArenaObject = ArenaObject.NONE;
    double closestDistance = Double.MAX_VALUE;
    if (lRacketCollision != null) {
      double distance = previousBallPosition.distance(lRacketCollision);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestCollidingArenaObject = ArenaObject.L_RACKET;
      }
    }
    if (rRacketCollision != null) {
      double distance = previousBallPosition.distance(rRacketCollision);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestCollidingArenaObject = ArenaObject.R_RACKET;
      }
    }
    if (arenaHorizontalEdgesCollision != null) {
      double distance = previousBallPosition.distance(arenaHorizontalEdgesCollision);
      if (distance < closestDistance) {
        closestDistance = distance;
        closestCollidingArenaObject = ArenaObject.HORIZONTAL_EDGES;
      }
    }
    switch (closestCollidingArenaObject) {
      case NONE:
        if (updatedBallState.position.x() < 0) {
          updatedLRacketState = updatedLRacketState.incrementScore();
        } else if (updatedBallState.position.x() > configuration.arenaXLength) {
          updatedRRacketState = updatedRRacketState.incrementScore();
        }
        break;
      case L_RACKET:

    }


    return null;
  }

  private Segment getAsSegment(BallState previousBallState, BallState nextBallState) {
    return new Segment(
        previousBallState.position,
        nextBallState.position
    );
  }

  private Segment getAsSegment(RacketState racketState) {
    if (racketState.side == Side.LEFT) {
      return new Segment(
          new Point(0, racketState.yCenter - configuration.racketsLength / 2),
          new Point(0, racketState.yCenter + configuration.racketsLength / 2));
    } else {
      return new Segment(
          new Point(configuration.arenaXLength, racketState.yCenter - configuration.racketsLength / 2),
          new Point(configuration.arenaXLength, racketState.yCenter + configuration.racketsLength / 2));
    }
  }

  private BallState updateBallState(BallState ballState, double deltaTime) {
    double deltaX = ballState.velocity.x() * deltaTime;
    double deltaY = ballState.velocity.y() * deltaTime;
    Point deltaPosition = new Point(deltaX, deltaY);
    return new BallState(ballState.position.sum(deltaPosition), ballState.velocity);
  }

  private RacketState updateRacketPosition(RacketState racketState, double deltaY) {
    double updatedY = racketState.yCenter + deltaY;
    updatedY = DoubleRange.UNIT.clip(updatedY / configuration.arenaYLength) * configuration.arenaYLength;
    return racketState.updatePosition(updatedY);
  }

  private void updateState(BallState ballState, RacketState lRacketState, RacketState rRacketState) {
    state = new State(
        configuration,
        lRacketState,
        rRacketState,
        ballState
    );
  }

  private Point randomBallVelocity() {
    double ballInitialAngle =
        configuration.ballInitialAngleRange.denormalize(configuration.randomGenerator.nextDouble());
    boolean flipBallVelocity = configuration.randomGenerator.nextBoolean();
    if (flipBallVelocity) {
      ballInitialAngle = ballInitialAngle + Math.PI;
    }
    double updatedBallInitialVelocity;
    try {
      updatedBallInitialVelocity = configuration.ballInitialVelocity
          * Math.pow(configuration.ballAccelerationRatePerPoint, (state.lRacketState.score() + state.rRacketState.score()));
      if (updatedBallInitialVelocity > configuration.ballMaxVelocity) {
        updatedBallInitialVelocity = configuration.ballMaxVelocity;
      }
    } catch (NullPointerException e) {
      updatedBallInitialVelocity = configuration.ballInitialVelocity;
    }
    double ballInitialXV = Math.cos(ballInitialAngle) * updatedBallInitialVelocity;
    double ballInitialYV = Math.sin(ballInitialAngle) * updatedBallInitialVelocity;
    return new Point(ballInitialXV, ballInitialYV);
  }
}
