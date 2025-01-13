package io.github.ericmedvet.jsdynsym.control.geometry;

import java.util.List;
import java.util.ArrayList;

public record Rectangle(Point topLeft, Point bottomRight) {

  public Rectangle {
    if (topLeft.x() >= bottomRight.x() || bottomRight.y() >= topLeft.y()) {
      throw new IllegalArgumentException("Invalid rectangle: Ensure bottomLeft is below and to the left of topRight.");
    }
  }

  // the list of intersection points returned sorted in ascending order based on the distance from segment.p1()
  public List<Point> intersection(Segment segment) {
    List<Point> intersections = new ArrayList<>();
    // check intersection with each edge
    for (Segment edge : List.of(topEdge(), bottomEdge(), leftEdge(), rightEdge())) {
      Point intersection = segment.intersection(edge);
      if (intersection != null) {
        intersections.add(intersection);
      }
    }
    intersections.sort((intersection1, intersection2) -> {
      double distance1 = segment.p1().distance(intersection1);
      double distance2 = segment.p1().distance(intersection2);
      return Double.compare(distance1, distance2);
    });
    return intersections;
  }

  public List<Point> verticalEdgesIntersections(Segment segment) {
    List<Point> verticalEdgesIntersections = new ArrayList<>();
    Point leftEdgeIntersection = segment.intersection(leftEdge());
    Point rightEdgeIntersection = segment.intersection(rightEdge());
    if (leftEdgeIntersection != null){
      verticalEdgesIntersections.add(leftEdgeIntersection);
    }
    if (rightEdgeIntersection != null){
      verticalEdgesIntersections.add(rightEdgeIntersection);
    }
    verticalEdgesIntersections.sort((intersection1, intersection2) -> {
      double distance1 = segment.p1().distance(intersection1);
      double distance2 = segment.p1().distance(intersection2);
      return Double.compare(distance1, distance2);
    });
    return verticalEdgesIntersections;
  }

  public List<Point> horizontalEdgesIntersections(Segment segment) {
    List<Point> horizontalEdgesIntersections = new ArrayList<>();
    Point topEdgeIntersection = segment.intersection(topEdge());
    Point bottomEdgeIntersection = segment.intersection(bottomEdge());
    if (topEdgeIntersection != null){
      horizontalEdgesIntersections.add(topEdgeIntersection);
    }
    if (bottomEdgeIntersection != null){
      horizontalEdgesIntersections.add(bottomEdgeIntersection);
    }
    horizontalEdgesIntersections.sort((intersection1, intersection2) -> {
      double distance1 = segment.p1().distance(intersection1);
      double distance2 = segment.p1().distance(intersection2);
      return Double.compare(distance1, distance2);
    });
    return horizontalEdgesIntersections;
  }

  public Segment leftEdge() {
    return new Segment(topLeft, bottomLeft());
  }

  public Segment rightEdge() {
    return new Segment(topRight(), bottomRight);
  }

  public Segment bottomEdge() {
    return new Segment(bottomLeft(), bottomRight);
  }

  public Segment topEdge() {
    return new Segment(topLeft, topRight());
  }

  public Point topRight() {
    return new Point(bottomRight.x(), topLeft.y());
  }

  public Point bottomLeft() {
    return new Point(topLeft.x(), bottomRight.y());
  }
}
