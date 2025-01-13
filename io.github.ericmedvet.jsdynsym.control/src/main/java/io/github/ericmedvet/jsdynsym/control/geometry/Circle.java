package io.github.ericmedvet.jsdynsym.control.geometry;

import java.util.ArrayList;
import java.util.List;

public record Circle(Point center, double radius) {

  public Circle {
    if (radius <= 0) {
      throw new IllegalArgumentException("Radius must be positive.");
    }
  }

  public boolean contains(Point point) {
    return center.distance(point) <= radius;
  }

  //TODO check
  // the list of intersection points returned sorted in ascending order based on the distance from segment.p1()
  public List<Point> intersection(Segment segment) {
    // translate the segment to the circle's coordinate system
    Point p1 = segment.p1();
    Point p2 = segment.p2();
    double dx = p2.x() - p1.x();
    double dy = p2.y() - p1.y();
    double fx = p1.x() - center.x();
    double fy = p1.y() - center.y();
    // quadratic equation coefficients
    double a = dx * dx + dy * dy;
    double b = 2 * (fx * dx + fy * dy);
    double c = fx * fx + fy * fy - radius * radius;
    // discriminant
    double discriminant = b * b - 4 * a * c;
    List<Point> intersections = new ArrayList<>();
    if (discriminant < 0) {
      // no intersection
      return intersections;
    }
    // calculate the t values for the segment line
    double sqrtDiscriminant = Math.sqrt(discriminant);
    double t1 = (-b - sqrtDiscriminant) / (2 * a);
    double t2 = (-b + sqrtDiscriminant) / (2 * a);
    // check if t1 lies on the segment
    if (t1 >= 0 && t1 <= 1) {
      double ix1 = p1.x() + t1 * dx;
      double iy1 = p1.y() + t1 * dy;
      intersections.add(new Point(ix1, iy1));
    }
    // check if t2 lies on the segment
    if (t2 >= 0 && t2 <= 1) {
      double ix2 = p1.x() + t2 * dx;
      double iy2 = p1.y() + t2 * dy;
      intersections.add(new Point(ix2, iy2));
    }
    return intersections;
  }

  @Override
  public String toString() {
    return "Circle[center=" + center + ", radius=" + radius + "]";
  }
}
